#include "vkr/planning/amcp_planner.hpp"
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/geometry/collision/collision_query.hpp"
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <tbb/task_group.h>
#include <chrono>
#include <fstream>
#include <algorithm>

namespace vkr
{
namespace planning
{

struct AMCPlanner::Implementation 
{
  VkrConfig config;
  
  std::shared_ptr<terrain::WaveletGrid> terrain;
  std::shared_ptr<geometry::collision::CollisionQuery> collision;
  std::shared_ptr<geometry::corridors::SpatialCorridorMap> corridors;
  std::shared_ptr<MultiResolutionGrid> grid;
  
  std::unique_ptr<search::ARAStarSearch> ara_star;
  std::unique_ptr<search::DStarLite> dstar_lite;
  
  tbb::concurrent_queue<geometry::corridors::CorridorFreeEvent> event_queue;
  
  std::atomic<bool> cancel_requested
  {
    false
  };
  tbb::task_group async_tasks;
  
  Stats stats
  {
  };
  std::chrono::steady_clock::time_point start_time;
  
  NodeId current_start = INVALID_NODE;
  NodeId current_goal = INVALID_NODE;
  UAVId current_uav = static_cast<UAVId>(INVALID_ID);
  bool has_initial_plan = false;
  
  geometry::BoundingBox cached_world_bounds;
  bool world_bounds_initialized = false;
  
  Implementation() 
  {
    start_time = std::chrono::steady_clock::now();
  }
  
  void initialize() 
  {
    if (!grid) 
    {
      grid = std::make_shared<MultiResolutionGrid>(config.planning);
    }
    
    if (!ara_star) 
    {
      ara_star = std::make_unique<search::ARAStarSearch>(config.planning);
      ara_star->setGrid(grid);
      setupSearchCallbacks();
    }
    
    if (!dstar_lite) 
    {
      dstar_lite = std::make_unique<search::DStarLite>(config.planning);
    }
  }
  
  geometry::BoundingBox determineWorldBounds(const PlanRequest& request) 
  {
    geometry::BoundingBox bounds;
    bool bounds_set = false;
    
    if (request.has_world_bounds) 
    {
      bounds = request.world_bounds;
      bounds_set = true;
    }
    
    if (terrain) 
    {
      geometry::BoundingBox terrain_bounds = terrain->getBounds();
      if (bounds_set) 
      {
        bounds.min = bounds.min.cwiseMin(terrain_bounds.min);
        bounds.max = bounds.max.cwiseMax(terrain_bounds.max);
      }
      else 
      {
        bounds = terrain_bounds;
        bounds_set = true;
      }
    }
    
    if (!bounds_set) 
    {
      bounds.min = request.start.cwiseMin(request.goal);
      bounds.max = request.start.cwiseMax(request.goal);
      
      float margin = std::max(200.0f, (bounds.max - bounds.min).maxCoeff() * 0.5f);
      bounds.min -= Eigen::Vector3f::Constant(margin);
      bounds.max += Eigen::Vector3f::Constant(margin);
    }
    
    float safety_margin = config.planning.coarse_grid_size * 2;
    bounds.min = bounds.min.cwiseMin(request.start - Eigen::Vector3f::Constant(safety_margin));
    bounds.min = bounds.min.cwiseMin(request.goal - Eigen::Vector3f::Constant(safety_margin));
    bounds.max = bounds.max.cwiseMax(request.start + Eigen::Vector3f::Constant(safety_margin));
    bounds.max = bounds.max.cwiseMax(request.goal + Eigen::Vector3f::Constant(safety_margin));
    
    Eigen::Vector3f size = bounds.max - bounds.min;
    float min_size = config.planning.coarse_grid_size * 10;
    for (int i = 0; i < 3; ++i) 
    {
      if (size[i] < min_size) 
      {
        float expand = (min_size - size[i]) * 0.5f;
        bounds.min[i] -= expand;
        bounds.max[i] += expand;
      }
    }
    
    return bounds;
  }
  
  void setupSearchCallbacks() 
  {
    ara_star->setCostFunction([this](NodeId from, NodeId to) 
    {
      Eigen::Vector3f from_pos = grid->getNodePosition(from, MultiResolutionGrid::Level::FINE);
      Eigen::Vector3f to_pos = grid->getNodePosition(to, MultiResolutionGrid::Level::FINE);
      return computeEdgeCost(from_pos, to_pos, current_uav);
    });
    
    ara_star->setValidityFunction([this](const Eigen::Vector3f& pos) 
    {
      return isValidPosition(pos);
    });
    
    ara_star->setHeuristicFunction(1, [this](const Eigen::Vector3f& from, const Eigen::Vector3f& to) 
    {
      if (!terrain) return 0.0f;
      
      float from_height = terrain->getHeight(from.x(), from.y());
      float to_height = terrain->getHeight(to.x(), to.y());
      
      float from_altitude = from.z() - from_height;
      float to_altitude = to.z() - to_height;
      
      const float safe_altitude = config.planning.safe_altitude;
      float altitude_penalty = 0.0f;
      
      if (from_altitude < safe_altitude) 
      {
        altitude_penalty += std::exp(-(from_altitude / safe_altitude)) * config.planning.height_penalty_weight;
      }
      
      if (to_altitude < safe_altitude) 
      {
        altitude_penalty += std::exp(-(to_altitude / safe_altitude)) * config.planning.height_penalty_weight;
      }
      
      float height_diff = to_height - from_height;
      if (height_diff > 0) 
      {
        float distance = (to - from).norm();
        float slope = (distance > 0.0f) ? height_diff / distance : 0.0f;
        altitude_penalty += slope * 2.0f * config.planning.height_penalty_weight;
      }
      
      return altitude_penalty;
    });
  }
  
  float computeEdgeCost(const Eigen::Vector3f& from, const Eigen::Vector3f& to, UAVId uav_id) 
  {
    float distance = (to - from).norm();
    float cost = distance;
    
    if (terrain) 
    {
      float from_height = terrain->getHeight(from.x(), from.y());
      float to_height = terrain->getHeight(to.x(), to.y());
      float altitude_from = from.z() - from_height;
      float altitude_to = to.z() - to_height;
      
      const float min_altitude = config.planning.min_altitude;
      if (altitude_from < min_altitude || altitude_to < min_altitude) 
      {
        cost += config.planning.height_penalty_weight * (min_altitude - std::min(altitude_from, altitude_to));
      }
    }
    
    if (collision) 
    {
      geometry::LineSegment segment
      {
        from, to
      };
      auto hit = collision->querySegment(segment, std::chrono::steady_clock::now(), uav_id);
      
      if (hit.hit) 
      {
        return INFINITY_F;
      }
    }
    
    return cost;
  }
  
  bool isValidPosition(const Eigen::Vector3f& pos) 
  {
    if (terrain) 
    {
      float ground_height = terrain->getHeight(pos.x(), pos.y());
      float altitude = pos.z() - ground_height;
      
      const float min_clearance = config.planning.min_clearance;
      if (altitude < min_clearance) 
      {
        return false;
      }
    }
    
    if (collision) 
    {
      float radius = config.geometry.corridor_radius;
      return collision->isFree(pos, radius, std::chrono::steady_clock::now(), current_uav);
    }
    
    return true;
  }
  
  bool isValidEdge(const Eigen::Vector3f& from, const Eigen::Vector3f& to) 
  {
    if (!collision) 
    {
      return true;
    }
    
    geometry::LineSegment segment
    {
      from, to
    };
    auto hit = collision->querySegment(segment, std::chrono::steady_clock::now(), current_uav);
    return !hit.hit;
  }
  
  float computeHeightPenalty(const Eigen::Vector3f& pos) 
  {
    if (!terrain) 
    {
      return 0.0f;
    }
    
    float ground_height = terrain->getHeight(pos.x(), pos.y());
    float altitude = pos.z() - ground_height;
    
    const float safe_altitude = config.planning.safe_altitude;
    if (altitude < safe_altitude) 
    {
      return std::exp(-(altitude / safe_altitude)) * config.planning.height_penalty_weight;
    }
    
    return 0.0f;
  }
  
  std::vector<Eigen::Vector3f> connectPathEndpoints(const std::vector<Eigen::Vector3f>& grid_path,
                                                     const Eigen::Vector3f& exact_start,
                                                     const Eigen::Vector3f& exact_goal) 
  {
    if (grid_path.empty()) 
    {
      return grid_path;
    }
    
    std::vector<Eigen::Vector3f> connected_path;
    connected_path.reserve(grid_path.size() + 2);
    
    if ((exact_start - grid_path.front()).norm() > GEOM_EPS) 
    {
      connected_path.push_back(exact_start);
    }
    
    connected_path.insert(connected_path.end(), grid_path.begin(), grid_path.end());
    
    if ((exact_goal - grid_path.back()).norm() > GEOM_EPS) 
    {
      connected_path.push_back(exact_goal);
    }
    
    return connected_path;
  }
};

AMCPlanner::AMCPlanner(const VkrConfig& config)
  : impl_(std::make_unique<Implementation>())
{
  impl_->config = config;
  impl_->initialize();
}

AMCPlanner::~AMCPlanner() 
{
  impl_->cancel_requested = true;
  impl_->async_tasks.wait();
}

void AMCPlanner::setTerrain(std::shared_ptr<terrain::WaveletGrid> terrain) 
{
  impl_->terrain = terrain;
  impl_->world_bounds_initialized = false;
}

void AMCPlanner::setCollisionQuery(std::shared_ptr<geometry::collision::CollisionQuery> collision) 
{
  impl_->collision = collision;
}

void AMCPlanner::setCorridors(std::shared_ptr<geometry::corridors::SpatialCorridorMap> corridors) 
{
  impl_->corridors = corridors;
}

PlanResult AMCPlanner::plan(const PlanRequest& request) 
{
  auto start_time = std::chrono::steady_clock::now();
  
  if (!request.isValid()) 
  {
    spdlog::error("AMCPlanner: Invalid plan request");
    PlanResult result;
    result.status = Status::FAILURE;
    return result;
  }
  
  if (!impl_->grid) 
  {
    impl_->initialize();
  }
  
  impl_->current_uav = request.uav_id;
  
  if (!isValidPosition(request.start) || !isValidPosition(request.goal))
  {
    spdlog::error("AMCPlanner: Start or goal position is not valid (e.g., inside an obstacle or too close to terrain)");
    PlanResult res;
    res.status = Status::FAILURE;
    return res;
  }
  
  geometry::BoundingBox world_bounds = impl_->determineWorldBounds(request);
  
  if (!impl_->world_bounds_initialized || 
      (impl_->cached_world_bounds.min - world_bounds.min).norm() > 100.0f ||
      (impl_->cached_world_bounds.max - world_bounds.max).norm() > 100.0f) 
  {
    spdlog::info("AMCPlanner: Initializing grid with bounds [{:.1f},{:.1f},{:.1f}] to [{:.1f},{:.1f},{:.1f}]",
                 world_bounds.min.x(), world_bounds.min.y(), world_bounds.min.z(),
                 world_bounds.max.x(), world_bounds.max.y(), world_bounds.max.z());
    
    impl_->grid->initialize(world_bounds);
    impl_->cached_world_bounds = world_bounds;
    impl_->world_bounds_initialized = true;
    
    impl_->has_initial_plan = false;
  }
  
  impl_->current_start = impl_->grid->getNode(request.start, MultiResolutionGrid::Level::FINE);
  impl_->current_goal = impl_->grid->getNode(request.goal, MultiResolutionGrid::Level::FINE);
  
  if (impl_->current_start == INVALID_NODE || impl_->current_goal == INVALID_NODE) 
  {
    spdlog::error("AMCPlanner: Start or goal position outside grid bounds");
    spdlog::debug("  Start: [{:.1f},{:.1f},{:.1f}], Goal: [{:.1f},{:.1f},{:.1f}]",
                  request.start.x(), request.start.y(), request.start.z(),
                  request.goal.x(), request.goal.y(), request.goal.z());
    spdlog::debug("  Grid bounds: [{:.1f},{:.1f},{:.1f}] to [{:.1f},{:.1f},{:.1f}]",
                  world_bounds.min.x(), world_bounds.min.y(), world_bounds.min.z(),
                  world_bounds.max.x(), world_bounds.max.y(), world_bounds.max.z());
    return PlanResult
    {
    };
  }
  
  Eigen::Vector3f grid_start = impl_->grid->getNodePosition(impl_->current_start, MultiResolutionGrid::Level::FINE);
  Eigen::Vector3f grid_goal = impl_->grid->getNodePosition(impl_->current_goal, MultiResolutionGrid::Level::FINE);
  
  float w0 = 1.0f;
  float w1 = impl_->terrain ? 0.1f : 0.0f;
  float w2 = 0.0f;
  
  impl_->ara_star->setHeuristicWeights(w0, w1, w2);
  
  processEvents();
  
  PlanRequest grid_request = request;
  grid_request.start = grid_start;
  grid_request.goal = grid_goal;
  
  PlanResult result = impl_->ara_star->search(grid_request);
  
  impl_->stats.total_plans++;
  
  if (result.isSuccess()) 
  {
    result.path = impl_->connectPathEndpoints(result.path, request.start, request.goal);
    
    if (!impl_->has_initial_plan) 
    {
      impl_->dstar_lite->initialize(impl_->current_start, impl_->current_goal, impl_->grid);
      impl_->has_initial_plan = true;
    }
    
    impl_->stats.successful_plans++;
    
    auto elapsed = std::chrono::steady_clock::now() - start_time;
    float elapsed_ms = std::chrono::duration<float, std::milli>(elapsed).count();
    impl_->stats.average_planning_time_ms = 
      (impl_->stats.average_planning_time_ms * (impl_->stats.total_plans - 1) + elapsed_ms) / 
      impl_->stats.total_plans;
    
    result.path_length = 0.0f;
    for (size_t i = 1; i < result.path.size(); ++i) 
    {
      result.path_length += (result.path[i] - result.path[i-1]).norm();
    }
    
    impl_->stats.average_path_length = 
      (impl_->stats.average_path_length * (impl_->stats.successful_plans - 1) + result.path_length) / 
      impl_->stats.successful_plans;
  }
  else 
  {
    spdlog::warn("AMCPlanner: Failed to find path (status: {})", static_cast<int>(result.status));
  }
  
  auto ara_stats = impl_->ara_star->getStats();
  impl_->stats.current_open_list_size = ara_stats.open_list_size;
  impl_->stats.current_epsilon = ara_stats.current_epsilon;
  
  return result;
}

std::future<PlanResult> AMCPlanner::planAsync(const PlanRequest& request) 
{
  impl_->cancel_requested = false;
  
  auto promise = std::make_shared<std::promise<PlanResult>>();
  auto future = promise->get_future();
  
  impl_->async_tasks.run([this, request, promise]() 
  {
    try 
    {
      auto result = plan(request);
      promise->set_value(result);
    }
    catch (...) 
    {
      promise->set_exception(std::current_exception());
    }
  });
  
  return future;
}

bool AMCPlanner::cancelAsync() 
{
  impl_->cancel_requested = true;
  impl_->async_tasks.cancel();
  impl_->async_tasks.wait();
  return true;
}

void AMCPlanner::pushEvent(const geometry::corridors::CorridorFreeEvent& event) 
{
  impl_->event_queue.push(event);
}

void AMCPlanner::updateEdgeCost(NodeId from, NodeId to, float new_cost) 
{
  if (impl_->has_initial_plan && impl_->dstar_lite) 
  {
    search::DStarLite::EdgeUpdate update;
    update.from = from;
    update.to = to;
    update.new_cost = new_cost;
    impl_->dstar_lite->updateEdgeCost(update);
  }
}

void AMCPlanner::clear() 
{
  impl_->ara_star.reset();
  impl_->dstar_lite.reset();
  impl_->grid.reset();
  impl_->has_initial_plan = false;
  impl_->world_bounds_initialized = false;
  impl_->stats = Stats
  {
  };
  impl_->initialize();
}

AMCPlanner::Stats AMCPlanner::getStats() const 
{
  return impl_->stats;
}

void AMCPlanner::processEvents() 
{
  geometry::corridors::CorridorFreeEvent event;
  std::vector<search::DStarLite::EdgeUpdate> updates;
  
  while (impl_->event_queue.try_pop(event)) 
  {
    spdlog::debug("AMCPlanner: Processing corridor free event for corridor {}, segment {}", 
                  event.corridor_id, event.segment_id);
    
    if (impl_->corridors && impl_->grid) 
    {
      Eigen::Vector3f current_pos = impl_->grid->getNodePosition(impl_->current_start, 
                                                                MultiResolutionGrid::Level::FINE);
      float region_size = impl_->config.geometry.corridor_radius * 4.0f;
      
      geometry::BoundingBox affected_region;
      affected_region.min = current_pos - Eigen::Vector3f::Constant(region_size);
      affected_region.max = current_pos + Eigen::Vector3f::Constant(region_size);
      
      geometry::corridors::Corridor corridor_info;
      if (impl_->corridors->getCorridorInfo(event.corridor_id, corridor_info)) 
      {
        float corridor_radius = corridor_info.radius;
        float grid_resolution = impl_->grid->getResolution(MultiResolutionGrid::Level::FINE);
        
        float margin = grid_resolution * 5.0f;
        affected_region.expand(Eigen::Vector3f::Constant(corridor_radius + margin));
      }
      
      const float resolution = impl_->grid->getResolution(MultiResolutionGrid::Level::FINE);
      
      for (float x = affected_region.min.x(); x <= affected_region.max.x(); x += resolution) 
      {
        for (float y = affected_region.min.y(); y <= affected_region.max.y(); y += resolution) 
        {
          for (float z = affected_region.min.z(); z <= affected_region.max.z(); z += resolution) 
          {
            Eigen::Vector3f pos(x, y, z);
            NodeId node_id = impl_->grid->getNode(pos, MultiResolutionGrid::Level::FINE);
            
            if (node_id == INVALID_NODE) continue;
            
            auto neighbors = impl_->grid->getNeighbors(node_id, MultiResolutionGrid::Level::FINE);
            
            for (NodeId neighbor_id : neighbors) 
            {
              Eigen::Vector3f neighbor_pos = impl_->grid->getNodePosition(neighbor_id, 
                                                                          MultiResolutionGrid::Level::FINE);
              
              float old_cost = impl_->grid->getEdgeCost(node_id, neighbor_id, 
                                                        MultiResolutionGrid::Level::FINE);
              float new_cost = impl_->computeEdgeCost(pos, neighbor_pos, impl_->current_uav);
              
              if (std::abs(new_cost - old_cost) > GEOM_EPS) 
              {
                search::DStarLite::EdgeUpdate update;
                update.from = node_id;
                update.to = neighbor_id;
                update.new_cost = new_cost;
                update.affected_region = affected_region;
                updates.push_back(update);
                
                impl_->grid->updateEdgeCost(node_id, neighbor_id, new_cost, 
                                          MultiResolutionGrid::Level::FINE);
              }
            }
          }
        }
      }
    }
  }
  
  if (!updates.empty() && impl_->has_initial_plan && impl_->dstar_lite) 
  {
    impl_->dstar_lite->batchUpdateEdgeCosts(updates);
    
    if (impl_->dstar_lite->needsReplan()) 
    {
      bool success = impl_->dstar_lite->replan();
      if (success) 
      {
        spdlog::info("AMCPlanner: Successfully replanned after {} edge updates", updates.size());
      }
    }
  }
}

float AMCPlanner::computeEdgeCost(const Eigen::Vector3f& from, 
                                 const Eigen::Vector3f& to,
                                 UAVId uav_id) const 
{
  return impl_->computeEdgeCost(from, to, uav_id);
}

float AMCPlanner::computeHeightPenalty(const Eigen::Vector3f& pos) const 
{
  return impl_->computeHeightPenalty(pos);
}

bool AMCPlanner::isValidPosition(const Eigen::Vector3f& pos) const 
{
  return impl_->isValidPosition(pos);
}

bool AMCPlanner::isValidEdge(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const 
{
  return impl_->isValidEdge(from, to);
}

void AMCPlanner::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["config"]["initial_epsilon"] = impl_->config.planning.initial_epsilon;
  j["config"]["height_penalty_weight"] = impl_->config.planning.height_penalty_weight;
  j["config"]["corridor_radius"] = impl_->config.geometry.corridor_radius;
  
  j["stats"]["total_plans"] = impl_->stats.total_plans;
  j["stats"]["successful_plans"] = impl_->stats.successful_plans;
  j["stats"]["average_planning_time_ms"] = impl_->stats.average_planning_time_ms;
  j["stats"]["average_path_length"] = impl_->stats.average_path_length;
  j["stats"]["current_epsilon"] = impl_->stats.current_epsilon;
  
  j["components"]["terrain"] = impl_->terrain != nullptr;
  j["components"]["collision"] = impl_->collision != nullptr;
  j["components"]["corridors"] = impl_->corridors != nullptr;
  
  j["state"]["has_initial_plan"] = impl_->has_initial_plan;
  j["state"]["current_start"] = impl_->current_start;
  j["state"]["current_goal"] = impl_->current_goal;
  j["state"]["current_uav"] = impl_->current_uav;
  j["state"]["world_bounds_initialized"] = impl_->world_bounds_initialized;
  
  if (impl_->world_bounds_initialized) 
  {
    j["world_bounds"]["min"] = 
    {
      impl_->cached_world_bounds.min.x(),
      impl_->cached_world_bounds.min.y(),
      impl_->cached_world_bounds.min.z()
    };
    j["world_bounds"]["max"] = 
    {
      impl_->cached_world_bounds.max.x(),
      impl_->cached_world_bounds.max.y(),
      impl_->cached_world_bounds.max.z()
    };
  }
  
  auto uptime = std::chrono::steady_clock::now() - impl_->start_time;
  j["runtime"]["uptime_seconds"] = std::chrono::duration<float>(uptime).count();
  
  std::ofstream ofs(filename);
  ofs << j.dump(2);
}

}
}