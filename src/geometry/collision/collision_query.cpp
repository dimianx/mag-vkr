#include "vkr/geometry/collision/collision_query.hpp"
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/geometry/intersections.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace vkr
{
namespace geometry
{
namespace collision
{

struct CollisionQuery::Implementation 
{
  GeometryConfig config;
  
  std::shared_ptr<terrain::WaveletGrid> terrain;
  std::shared_ptr<static_obstacles::BoundingSphereTree> static_obstacles;
  std::shared_ptr<corridors::SpatialCorridorMap> corridors;
  
  std::unique_ptr<UniformGrid> static_grid;
  
  bool early_exit_enabled = true;
  
  mutable PerformanceStats stats;
  mutable std::mutex stats_mutex;
  mutable std::chrono::steady_clock::time_point last_query_start;
  
  Implementation(const GeometryConfig& cfg) 
    : config(cfg)
  {
    spdlog::debug("CollisionQuery initialized with grid_cell_size={}, hash_table_size={}", 
                  config.grid_cell_size, config.hash_table_size);
  }
  
  void initializeStaticGrid() 
  {
    if (!static_obstacles) return;
    
    auto stats = static_obstacles->getStats();
    if (stats.total_nodes == 0) return;
    
    BoundingBox world_bounds;
    if (terrain) 
    {
      world_bounds = terrain->getBounds();
      world_bounds.min.z() -= 100.0f;
      world_bounds.max.z() += 500.0f;
    }
    else 
    {
      world_bounds.min = Eigen::Vector3f(-1000, -1000, -100);
      world_bounds.max = Eigen::Vector3f(1000, 1000, 500);
    }
    
    static_grid = std::make_unique<UniformGrid>(
      config.grid_cell_size,
      world_bounds,
      config.hash_table_size
    );
    
    static_grid->insert(0, world_bounds);
    
    spdlog::info("Initialized static obstacle spatial grid:");
    spdlog::info("  - World bounds: [{:.1f}, {:.1f}, {:.1f}] to [{:.1f}, {:.1f}, {:.1f}]",
                 world_bounds.min.x(), world_bounds.min.y(), world_bounds.min.z(),
                 world_bounds.max.x(), world_bounds.max.y(), world_bounds.max.z());
    spdlog::info("  - Grid cell size: {:.1f}m", config.grid_cell_size);
    spdlog::info("  - Hash table size: {}", config.hash_table_size);
    spdlog::debug("  - BST stats: {} nodes, {} leaves, depth {}", 
                  stats.total_nodes, stats.leaf_nodes, stats.max_depth);
  }
  
  HitResult checkTerrain(const Capsule& capsule) const 
  {
    HitResult result;
    if (!terrain) return result;
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.terrain_checks++;
    }
    
    auto terrain_bounds = terrain->getBounds();
    
    Eigen::Vector3f axis = capsule.p1 - capsule.p0;
    float capsule_length = axis.norm();
    if (capsule_length > GEOM_EPS) 
    {
      axis.normalize();
    } 
    else 
    {
      axis = Eigen::Vector3f(0, 0, 1);
      capsule_length = 0;
    }
    
    float min_x = std::min(capsule.p0.x(), capsule.p1.x()) - capsule.radius;
    float max_x = std::max(capsule.p0.x(), capsule.p1.x()) + capsule.radius;
    float min_y = std::min(capsule.p0.y(), capsule.p1.y()) - capsule.radius;
    float max_y = std::max(capsule.p0.y(), capsule.p1.y()) + capsule.radius;
    
    min_x = std::max(min_x, terrain_bounds.min.x());
    max_x = std::min(max_x, terrain_bounds.max.x());
    min_y = std::max(min_y, terrain_bounds.min.y());
    max_y = std::min(max_y, terrain_bounds.max.y());
    
    if (min_x > max_x || min_y > max_y) 
    {
      return result;
    }
    
    const float sample_spacing = 1.0f;
    int samples_x = std::max(1, static_cast<int>((max_x - min_x) / sample_spacing) + 1);
    int samples_y = std::max(1, static_cast<int>((max_y - min_y) / sample_spacing) + 1);
    
    bool collision_found = false;
    float min_penetration = std::numeric_limits<float>::max();
    Eigen::Vector3f closest_terrain_point;
    Eigen::Vector3f closest_capsule_point;
    
    for (int ix = 0; ix < samples_x; ++ix) 
    {
      for (int iy = 0; iy < samples_y; ++iy) 
      {
        float x = min_x + (max_x - min_x) * ix / std::max(1, samples_x - 1);
        float y = min_y + (max_y - min_y) * iy / std::max(1, samples_y - 1);
        
        Eigen::Vector3f ground_point(x, y, 0);
        
        float t = 0.0f;
        if (capsule_length > GEOM_EPS) 
        {
          Eigen::Vector3f to_point = ground_point - Eigen::Vector3f(capsule.p0.x(), capsule.p0.y(), 0);
          t = std::clamp(axis.head<2>().dot(to_point.head<2>()) / capsule_length, 0.0f, 1.0f);
        }
        
        Eigen::Vector3f axis_point = capsule.p0 + t * capsule_length * axis;
        float xy_distance = (ground_point - Eigen::Vector3f(axis_point.x(), axis_point.y(), 0)).head<2>().norm();
        
        if (xy_distance <= capsule.radius) 
        {
          float terrain_height = terrain->getHeight(x, y);
          
          float vertical_extent = std::sqrt(std::max(0.0f, capsule.radius * capsule.radius - xy_distance * xy_distance));
          float capsule_bottom = axis_point.z() - vertical_extent;
          
          float penetration = terrain_height - capsule_bottom;
          
          if (penetration > 0) 
          {
            collision_found = true;
            if (penetration < min_penetration) 
            {
              min_penetration = penetration;
              closest_terrain_point = Eigen::Vector3f(x, y, terrain_height);
              closest_capsule_point = Eigen::Vector3f(x, y, capsule_bottom);
            }
          }
        }
      }
    }
    
    if (collision_found) 
    {
      result.hit = true;
      result.t_min = 0.0f;
      result.object_type = HitResult::ObjectType::TERRAIN;
      result.hit_point = closest_terrain_point;
      
      float h = 1.0f;
      float x = closest_terrain_point.x();
      float y = closest_terrain_point.y();
      
      float x_minus = std::max(x - h, terrain_bounds.min.x());
      float x_plus = std::min(x + h, terrain_bounds.max.x());
      float y_minus = std::max(y - h, terrain_bounds.min.y());
      float y_plus = std::min(y + h, terrain_bounds.max.y());
      
      float h_x0 = terrain->getHeight(x_minus, y);
      float h_x1 = terrain->getHeight(x_plus, y);
      float h_y0 = terrain->getHeight(x, y_minus);
      float h_y1 = terrain->getHeight(x, y_plus);
      
      float dx = x_plus - x_minus;
      float dy = y_plus - y_minus;
      
      Eigen::Vector3f normal(
        -(h_x1 - h_x0) / dx,
        -(h_y1 - h_y0) / dy,
        1.0f
      );
      result.hit_normal = normal.normalized();
    }
    
    return result;
  }
  
  HitResult checkStaticObstacles(const Capsule& capsule) const 
  {
    HitResult result;
    if (!static_obstacles) return result;
    
    if (static_grid) 
    {
      BoundingBox capsule_bbox;
      capsule_bbox.min = capsule.p0.cwiseMin(capsule.p1) - Eigen::Vector3f::Constant(capsule.radius);
      capsule_bbox.max = capsule.p0.cwiseMax(capsule.p1) + Eigen::Vector3f::Constant(capsule.radius);
      
      auto candidates = static_grid->query(capsule_bbox);
      
      {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.broad_phase_checks += candidates.size();
      }
      
      if (candidates.empty()) 
      {
        return result;
      }
    }
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.narrow_phase_checks++;
      stats.static_checks++;
    }
    
    static_obstacles->intersectCapsule(capsule, result);
    
    if (result.hit) 
    {
      result.object_type = HitResult::ObjectType::STATIC_OBSTACLE;
    }
    
    return result;
  }
  
  HitResult checkCorridors(const Capsule& capsule, UAVId uav_id) const 
  {
    HitResult result;
    if (!corridors) return result;
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.corridor_checks++;
    }
    
    if (corridors->intersectCapsule(capsule, uav_id)) 
    {
      result.hit = true;
      result.t_min = 0.0f;
      result.object_type = HitResult::ObjectType::CORRIDOR;
      
      Eigen::Vector3f direction = capsule.p1 - capsule.p0;
      float length = direction.norm();
      if (length > GEOM_EPS) 
      {
        direction.normalize();
        
        const int num_samples = 10;
        for (int i = 0; i <= num_samples; ++i) 
        {
          float t = static_cast<float>(i) / num_samples;
          Eigen::Vector3f point = capsule.p0 + t * length * direction;
          
          if (corridors->isInsideOtherCorridor(point, uav_id)) 
          {
            result.hit_point = point;
            
            float delta = 1.0f;
            Eigen::Vector3f gradient(0, 0, 0);
            
            for (int axis = 0; axis < 3; ++axis) 
            {
              Eigen::Vector3f offset = Eigen::Vector3f::Zero();
              offset[axis] = delta;
              
              bool inside_plus = corridors->isInsideOtherCorridor(point + offset, uav_id);
              bool inside_minus = corridors->isInsideOtherCorridor(point - offset, uav_id);
              
              if (inside_plus && !inside_minus) 
              {
                gradient[axis] = -1.0f;
              }
              else if (!inside_plus && inside_minus) 
              {
                gradient[axis] = 1.0f;
              }
            }
            
            if (gradient.norm() > GEOM_EPS) 
            {
              result.hit_normal = gradient.normalized();
            }
            else 
            {
              Eigen::Vector3f capsule_center = (capsule.p0 + capsule.p1) * 0.5f;
              result.hit_normal = (capsule_center - point).normalized();
            }
            
            break;
          }
        }
      }
      else 
      {
        result.hit_point = capsule.p0;
        result.hit_normal = Eigen::Vector3f(0, 0, 1);
      }
    }
    
    return result;
  }
};

CollisionQuery::CollisionQuery(const GeometryConfig& config)
  : impl_(std::make_unique<Implementation>(config)) 
{
  math::simd::logSIMDInfo();
  spdlog::set_level(spdlog::level::debug);
}

CollisionQuery::~CollisionQuery() = default;

void CollisionQuery::setTerrain(std::shared_ptr<terrain::WaveletGrid> terrain) 
{
  impl_->terrain = terrain;
  spdlog::info("CollisionQuery: Terrain component set");
}

void CollisionQuery::setStaticObstacles(std::shared_ptr<static_obstacles::BoundingSphereTree> static_obs) 
{
  impl_->static_obstacles = static_obs;
  impl_->initializeStaticGrid();
  spdlog::info("CollisionQuery: Static obstacles component set");
}

void CollisionQuery::setCorridors(std::shared_ptr<corridors::SpatialCorridorMap> corridors) 
{
  impl_->corridors = corridors;
  spdlog::info("CollisionQuery: Corridors component set");
}

HitResult CollisionQuery::queryCapsule(const Capsule& capsule,
                                      const std::chrono::steady_clock::time_point& time,
                                      UAVId uav_id) const 
{
  auto start_time = std::chrono::steady_clock::now();
  impl_->last_query_start = start_time;
  
  HitResult result;
  
  auto terrain_result = impl_->checkTerrain(capsule);
  if (terrain_result.hit) 
  {
    result = terrain_result;
    spdlog::debug("Terrain collision detected at height {:.2f}", terrain_result.hit_point.z());
    
    if (impl_->early_exit_enabled) 
    {
      std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
      impl_->stats.last_query_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_time);
      return result;
    }
  }
  
  if (uav_id != static_cast<UAVId>(INVALID_ID)) 
  {
    auto corridor_result = impl_->checkCorridors(capsule, uav_id);
    if (corridor_result.hit) 
    {
      spdlog::debug("Corridor violation detected for UAV {}", uav_id);
      
      if (!result.hit || corridor_result.t_min < result.t_min) 
      {
        result = corridor_result;
      }
      
      if (impl_->early_exit_enabled) 
      {
        std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
        impl_->stats.last_query_time = 
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
      }
    }
  }
  
  auto static_result = impl_->checkStaticObstacles(capsule);
  if (static_result.hit) 
  {
    spdlog::debug("Static obstacle collision detected, object_id: {}", static_result.object_id);
    
    if (!result.hit || static_result.t_min < result.t_min) 
    {
      result = static_result;
    }
  }
  
  std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
  impl_->stats.last_query_time = 
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  
  spdlog::debug("Query completed in {}us, hit: {}", 
                impl_->stats.last_query_time.count(), result.hit);
  
  return result;
}

HitResult CollisionQuery::queryCapsule(const Capsule& capsule,
                                      const std::chrono::steady_clock::time_point& t0,
                                      const std::chrono::steady_clock::time_point& t1,
                                      UAVId uav_id) const 
{
  return queryCapsule(capsule, t0, uav_id);
}

HitResult CollisionQuery::querySegment(const LineSegment& segment,
                                      const std::chrono::steady_clock::time_point& time,
                                      UAVId uav_id) const 
{
  Capsule capsule;
  capsule.p0 = segment.start;
  capsule.p1 = segment.end;
  capsule.radius = 0.0f;
  
  return queryCapsule(capsule, time, uav_id);
}

HitResult CollisionQuery::querySegment(const LineSegment& segment,
                                      const std::chrono::steady_clock::time_point& t0,
                                      const std::chrono::steady_clock::time_point& t1,
                                      UAVId uav_id) const 
{
  Capsule capsule;
  capsule.p0 = segment.start;
  capsule.p1 = segment.end;
  capsule.radius = 0.0f;
  
  return queryCapsule(capsule, t0, t1, uav_id);
}

HitResult CollisionQuery::queryPath(const std::vector<Eigen::Vector3f>& path,
                                   float radius,
                                   const std::chrono::steady_clock::time_point& start_time,
                                   float velocity,
                                   UAVId uav_id) const 
{
  HitResult result;
  
  if (path.size() < 2 || velocity <= 0) 
  {
    return result;
  }
  
  auto current_time = start_time;
  
  for (size_t i = 0; i < path.size() - 1; ++i) 
  {
    Capsule segment_capsule;
    segment_capsule.p0 = path[i];
    segment_capsule.p1 = path[i + 1];
    segment_capsule.radius = radius;
    
    float segment_length = (path[i + 1] - path[i]).norm();
    auto segment_duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<float>(segment_length / velocity));
    auto segment_end_time = current_time + segment_duration;
    
    auto segment_result = queryCapsule(segment_capsule, current_time, segment_end_time, uav_id);
    
    if (segment_result.hit) 
    {
      result = segment_result;
      float path_progress = 0.0f;
      for (size_t j = 0; j < i; ++j) 
      {
        path_progress += (path[j + 1] - path[j]).norm();
      }
      path_progress += segment_result.t_min * segment_length;
      
      float total_length = 0.0f;
      for (size_t j = 0; j < path.size() - 1; ++j) 
      {
        total_length += (path[j + 1] - path[j]).norm();
      }
      
      result.t_min = path_progress / total_length;
      break;
    }
    
    current_time = segment_end_time;
  }
  
  return result;
}

bool CollisionQuery::isFree(const Eigen::Vector3f& position,
                           float radius,
                           const std::chrono::steady_clock::time_point& time,
                           UAVId uav_id) const 
{
  Sphere sphere;
  sphere.center = position;
  sphere.radius = radius;
  
  Capsule capsule;
  capsule.p0 = position;
  capsule.p1 = position;
  capsule.radius = radius;
  
  auto result = queryCapsule(capsule, time, uav_id);
  return !result.hit;
}

float CollisionQuery::getDistance(const Eigen::Vector3f& position,
                                 const std::chrono::steady_clock::time_point& time,
                                 UAVId uav_id) const 
{
  float min_distance = std::numeric_limits<float>::max();
  
  if (impl_->terrain) 
  {
    float terrain_height = impl_->terrain->getHeight(position.x(), position.y());
    float terrain_distance = position.z() - terrain_height;
    min_distance = std::min(min_distance, terrain_distance);
  }
  
  if (impl_->static_obstacles) 
  {
    if (impl_->static_obstacles->isPointInside(position))
    {
      min_distance = std::min(min_distance, 0.0f);
    }
    else
    {
      float static_distance = impl_->static_obstacles->getDistance(position);
      min_distance = std::min(min_distance, static_distance);
    }
  }
  
  if (impl_->corridors && uav_id != static_cast<UAVId>(INVALID_ID)) 
  {
    float corridor_distance = impl_->corridors->getDistanceToBoundary(position, uav_id);
    min_distance = std::min(min_distance, corridor_distance);
  }
  
  return min_distance;
}

void CollisionQuery::batchQueryCapsule(const Capsule* capsules,
                                      size_t count,
                                      const std::chrono::steady_clock::time_point& time,
                                      HitResult* results,
                                      UAVId uav_id) const 
{
  namespace simd = math::simd;
  
  if (count > 1000) 
  {
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, count),
      [&](const tbb::blocked_range<size_t>& range) 
      {
        for (size_t i = range.begin(); i != range.end(); ++i) 
        {
          results[i] = queryCapsule(capsules[i], time, uav_id);
        }
      }
    );
  }
  else 
  {
    simd::simdLoop(count, [&](size_t i, bool) 
    {
      results[i] = queryCapsule(capsules[i], time, uav_id);
    });
  }
}

std::vector<CollisionQuery::ObstacleInfo> CollisionQuery::getObstaclesInRadius(
    const Eigen::Vector3f& position,
    float radius,
    const std::chrono::steady_clock::time_point& time) const 
{
  std::vector<ObstacleInfo> obstacles;
  
  BoundingBox query_box;
  query_box.min = position - Eigen::Vector3f::Constant(radius);
  query_box.max = position + Eigen::Vector3f::Constant(radius);
  
  if (impl_->static_grid) 
  {
    auto candidates = impl_->static_grid->query(query_box);
    for (ObjectId id : candidates) 
    {
      ObstacleInfo info;
      info.id = id;
      info.type = HitResult::ObjectType::STATIC_OBSTACLE;
      info.distance = impl_->static_obstacles->getDistance(position);
      info.closest_point = position;
      info.bbox = query_box;
      obstacles.push_back(info);
    }
  }
  
  return obstacles;
}

CollisionQuery::PerformanceStats CollisionQuery::getPerformanceStats() const 
{
  std::lock_guard<std::mutex> lock(impl_->stats_mutex);
  return impl_->stats;
}

void CollisionQuery::resetPerformanceStats() 
{
  std::lock_guard<std::mutex> lock(impl_->stats_mutex);
  impl_->stats = PerformanceStats();
}

void CollisionQuery::setEarlyExitEnabled(bool enabled) 
{
  impl_->early_exit_enabled = enabled;
}

void CollisionQuery::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["components"] = {
    {"terrain", impl_->terrain != nullptr},
    {"static_obstacles", impl_->static_obstacles != nullptr},
    {"corridors", impl_->corridors != nullptr}
  };
  
  j["config"] = {
    {"grid_cell_size", impl_->config.grid_cell_size},
    {"hash_table_size", impl_->config.hash_table_size},
    {"early_exit_enabled", impl_->early_exit_enabled}
  };
  
  auto stats = getPerformanceStats();
  j["performance"] = {
    {"broad_phase_checks", stats.broad_phase_checks},
    {"narrow_phase_checks", stats.narrow_phase_checks},
    {"terrain_checks", stats.terrain_checks},
    {"static_checks", stats.static_checks},
    {"corridor_checks", stats.corridor_checks},
    {"last_query_time_us", stats.last_query_time.count()}
  };
  
  if (impl_->static_obstacles) 
  {
    auto bst_stats = impl_->static_obstacles->getStats();
    j["static_obstacles_stats"] = {
      {"total_nodes", bst_stats.total_nodes},
      {"leaf_nodes", bst_stats.leaf_nodes},
      {"max_depth", bst_stats.max_depth},
      {"avg_faces_per_leaf", bst_stats.average_faces_per_leaf}
    };
  }
  
  if (impl_->corridors) 
  {
    auto corridor_stats = impl_->corridors->getStats();
    j["corridors_stats"] = {
      {"total_corridors", corridor_stats.total_corridors},
      {"total_segments", corridor_stats.total_segments},
      {"active_segments", corridor_stats.active_segments},
      {"freed_segments", corridor_stats.freed_segments}
    };
  }
  
  if (impl_->static_grid) 
  {
    auto grid_stats = impl_->static_grid->getStats();
    j["static_grid_stats"] = {
      {"total_objects", grid_stats.total_objects},
      {"occupied_cells", grid_stats.occupied_cells},
      {"avg_objects_per_cell", grid_stats.average_objects_per_cell},
      {"max_objects_in_cell", grid_stats.max_objects_in_cell}
    };
  }
  
  std::ofstream file(filename);
  file << j.dump(2);
  spdlog::info("Exported collision query state to {}", filename);
}

}
}
}