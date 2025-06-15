#include "vkr/planning/search/ara_star.hpp"
#include "vkr/math/simd/batch_operations.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <algorithm>

namespace vkr
{
namespace planning
{
namespace search
{

struct ARAStarSearch::Implementation 
{
  PlanningConfig config;
  std::shared_ptr<MultiResolutionGrid> grid;
  
  struct LevelState 
  {
    std::unordered_map<NodeId, SearchNode> nodes;
    std::unordered_set<NodeId> closed_set;
    std::unordered_set<NodeId> inconsistent_set;
  };
  std::array<LevelState, 3> level_states;
  
  OpenList open_list;
  
  float epsilon = 2.5f;
  float epsilon_prime = 2.5f;
  NodeId start_node = INVALID_NODE;
  NodeId goal_node = INVALID_NODE;
  Eigen::Vector3f goal_position;
  MultiResolutionGrid::Level current_level = MultiResolutionGrid::Level::FINE;
  
  float w0 = 1.0f;
  float w1 = 0.0f;
  float w2 = 0.0f;
  
  std::array<HeuristicFunction, 3> heuristic_functions;
  CostFunction cost_function;
  ValidityFunction validity_function;
  
  Stats stats
  {
  };
  
  PlanResult current_solution;
  std::chrono::steady_clock::time_point search_start_time;
  
  float coarse_search_time_ratio = 0.2f;
  float fine_search_time_ratio = 0.7f;
  float ultra_search_time_ratio = 0.1f;
  
  void reset() 
  {
    for (auto& level_state : level_states) 
    {
      level_state.nodes.clear();
      level_state.closed_set.clear();
      level_state.inconsistent_set.clear();
    }
    open_list.clear();
    stats = Stats
    {
    };
  }
  
  SearchNode& getNode(NodeId id, MultiResolutionGrid::Level level) 
  {
    auto& level_state = level_states[static_cast<size_t>(level)];
    auto it = level_state.nodes.find(id);
    if (it == level_state.nodes.end()) 
    {
      SearchNode node;
      node.id = id;
      level_state.nodes[id] = node;
      stats.nodes_generated++;
      return level_state.nodes[id];
    }
    return it->second;
  }
  
  void insertOpen(NodeId id) 
  {
    auto& node = getNode(id, current_level);
    if (!node.isOpen()) 
    {
      node.setOpen(true);
      node.setClosed(false);
      auto key = node.getKey(epsilon);
      open_list.push(
      {
        key, id
      });
    }
  }
  
  void updateKey(NodeId id) 
  {
    auto& node = getNode(id, current_level);
    if (node.isOpen()) 
    {
      auto key = node.getKey(epsilon);
      open_list.push(
      {
        key, id
      });
    }
  }
  
  float computeHeuristic(const Eigen::Vector3f& pos) 
  {
    SearchNode::HeuristicSet h;
    
    h.h0 = (pos - goal_position).norm();
    
    if (heuristic_functions[1]) 
    {
      h.h1 = heuristic_functions[1](pos, goal_position);
    }
    
    if (heuristic_functions[2]) 
    {
      h.h2 = heuristic_functions[2](pos, goal_position);
    }
    
    return h.getWeighted(w0, w1, w2);
  }
  
  float getEdgeCost(NodeId from, NodeId to) 
  {
    if (cost_function) 
    {
      return cost_function(from, to);
    }
    
    return grid->getEdgeCost(from, to, current_level);
  }
  
  bool isValid(const Eigen::Vector3f& pos) 
  {
    if (validity_function) 
    {
      return validity_function(pos);
    }
    return true;
  }
};

ARAStarSearch::ARAStarSearch(const PlanningConfig& config)
  : impl_(std::make_unique<Implementation>())
{
  impl_->config = config;
  impl_->epsilon = config.initial_epsilon;
  
  impl_->heuristic_functions[0] = [](const Eigen::Vector3f& from, const Eigen::Vector3f& to) 
  {
    return (to - from).norm();
  };
}

ARAStarSearch::~ARAStarSearch() = default;

void ARAStarSearch::setGrid(std::shared_ptr<MultiResolutionGrid> grid) 
{
  impl_->grid = grid;
}

void ARAStarSearch::setHeuristicWeights(float w0, float w1, float w2) 
{
  impl_->w0 = w0;
  impl_->w1 = w1;
  impl_->w2 = w2;
}

PlanResult ARAStarSearch::search(const PlanRequest& request) 
{
  if (!impl_->grid) 
  {
    spdlog::error("ARAStarSearch: Grid not set");
    return PlanResult
    {
    };
  }
  
  impl_->reset();
  
  impl_->search_start_time = std::chrono::steady_clock::now();
  impl_->goal_position = request.goal;
  
  auto elapsed = std::chrono::steady_clock::now() - impl_->search_start_time;
  if (std::chrono::duration<float, std::milli>(elapsed).count() >= request.max_runtime_ms) 
  {
    PlanResult result;
    result.status = Status::TIMEOUT;
    result.planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    return result;
  }
  
  PlanResult result;
  
  std::array<MultiResolutionGrid::Level, 3> search_levels = 
  {
    MultiResolutionGrid::Level::COARSE,
    MultiResolutionGrid::Level::FINE,
    MultiResolutionGrid::Level::ULTRA
  };
  
  std::array<float, 3> time_ratios = 
  {
    impl_->coarse_search_time_ratio,
    impl_->fine_search_time_ratio,
    impl_->ultra_search_time_ratio
  };
  
  std::vector<Eigen::Vector3f> best_path;
  float best_cost = INFINITY_F;
  bool final_solution_found = false;
  
  for (size_t level_idx = 0; level_idx < 2; ++level_idx)
  {
    auto level = search_levels[level_idx];
    impl_->current_level = level;
    
    float level_time_ms = request.max_runtime_ms * time_ratios[level_idx];
    auto level_deadline = impl_->search_start_time + 
                         std::chrono::milliseconds(static_cast<int>(level_time_ms));
    
    impl_->start_node = impl_->grid->getNode(request.start, level);
    impl_->goal_node = impl_->grid->getNode(request.goal, level);
    
    if (impl_->start_node == INVALID_NODE || impl_->goal_node == INVALID_NODE) 
    {
      continue;
    }
    
    auto& start = impl_->getNode(impl_->start_node, level);
    start.g_cost = 0.0f;
    start.h_cost = impl_->computeHeuristic(request.start);
    start.back_pointer = INVALID_NODE;
    impl_->insertOpen(impl_->start_node);
    
    bool solution_found_on_level = false;
    
    while (!impl_->open_list.empty() && std::chrono::steady_clock::now() < level_deadline) 
    {
      std::pair<SearchNode::Key, NodeId> top_element;
      if (!impl_->open_list.try_pop(top_element)) 
      {
        break;
      }
      
      auto [key, current_id] = top_element;
      auto& current = impl_->getNode(current_id, level);
      
      if (current.isClosed() || current.getKey(impl_->epsilon) < key) 
      {
        continue;
      }
      
      if (current_id == impl_->goal_node) 
      {
        solution_found_on_level = true;
        if (current.g_cost < best_cost) 
        {
          best_cost = current.g_cost;
          impl_->current_solution.cost = current.g_cost;
          impl_->current_solution.epsilon = impl_->epsilon;
        }
        break;
      }
      
      current.setOpen(false);
      current.setClosed(true);
      impl_->level_states[static_cast<size_t>(level)].closed_set.insert(current_id);
      impl_->stats.nodes_expanded++;
      
      expandNode(current_id);
    }
    
    if (solution_found_on_level) 
    {
      auto level_path = reconstructPath(impl_->goal_node);
      
      if (!level_path.empty()) 
      {
        std::vector<Eigen::Vector3f> connected_path;
        connected_path.reserve(level_path.size() + 2);
        
        if ((request.start - level_path.front()).norm() > GEOM_EPS) 
        {
          connected_path.push_back(request.start);
        }
        
        connected_path.insert(connected_path.end(), level_path.begin(), level_path.end());
        
        if ((request.goal - level_path.back()).norm() > GEOM_EPS) 
        {
          connected_path.push_back(request.goal);
        }
        
        level_path = connected_path;
      }
      
      best_path = level_path;
      best_cost = impl_->current_solution.cost;
      final_solution_found = true;
    }
    else
    {
      final_solution_found = false;
      best_path.clear();
      break;
    }
    
    impl_->open_list.clear();
  }
  
  elapsed = std::chrono::steady_clock::now() - impl_->search_start_time;
  result.planning_time = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
  result.nodes_expanded = impl_->stats.nodes_expanded;
  result.nodes_generated = impl_->stats.nodes_generated;
  
  if (final_solution_found) 
  {
    result.status = Status::SUCCESS;
    result.cost = best_cost;
    result.epsilon = impl_->epsilon;
    result.path = best_path;
    
    result.path_length = 0.0f;
    for (size_t i = 1; i < result.path.size(); ++i) 
    {
      result.path_length += (result.path[i] - result.path[i-1]).norm();
    }
    
    result.max_altitude = 0.0f;
    for (const auto& pos : result.path) 
    {
      result.max_altitude = std::max(result.max_altitude, pos.z());
    }
    
    impl_->current_solution = result;
  }
  else 
  {
    float elapsed_ms = std::chrono::duration<float, std::milli>(elapsed).count();
    if (elapsed_ms >= request.max_runtime_ms) 
    {
      result.status = Status::TIMEOUT;
    }
    else 
    {
      result.status = Status::FAILURE;
    }
    spdlog::info("ARAStarSearch: No solution found (expanded {} nodes)", impl_->stats.nodes_expanded);
  }
  
  impl_->stats.open_list_size = impl_->open_list.size();
  impl_->stats.current_epsilon = impl_->epsilon;
  impl_->stats.solution_cost = result.cost;
  
  return result;
}
      
MultiResolutionGrid::Level ARAStarSearch::getCurrentSearchLevel() const 
{
  return impl_->current_level;
}

bool ARAStarSearch::improveSolution(float new_epsilon, float time_limit_ms) 
{
  if (!impl_->current_solution.isSuccess()) 
  {
    return false;
  }
  
  if (new_epsilon >= impl_->epsilon) 
  {
    return false;
  }
  
  impl_->epsilon = new_epsilon;
  auto start_time = std::chrono::steady_clock::now();
  auto deadline = start_time + std::chrono::milliseconds(static_cast<int>(time_limit_ms));
  
  for (NodeId id : impl_->level_states[static_cast<size_t>(impl_->current_level)].inconsistent_set) 
  {
    auto& node = impl_->getNode(id, impl_->current_level);
    if (node.g_cost < INFINITY_F) 
    {
      impl_->insertOpen(id);
    }
  }
  impl_->level_states[static_cast<size_t>(impl_->current_level)].inconsistent_set.clear();
  
  std::vector<NodeId> open_nodes;
  std::pair<SearchNode::Key, NodeId> element;
  while (impl_->open_list.try_pop(element)) 
  {
    open_nodes.push_back(element.second);
  }
  
  for (NodeId id : open_nodes) 
  {
    impl_->insertOpen(id);
  }
  
  bool improved = false;
  
  while (!impl_->open_list.empty() && std::chrono::steady_clock::now() < deadline) 
  {
    std::pair<SearchNode::Key, NodeId> top_element;
    if (!impl_->open_list.try_pop(top_element)) 
    {
      break;
    }
    
    auto [key, current_id] = top_element;
    auto& current = impl_->getNode(current_id, impl_->current_level);
    
    if (current.isClosed() || current.getKey(impl_->epsilon) < key) 
    {
      continue;
    }
    
    if (current_id == impl_->goal_node && current.g_cost < impl_->current_solution.cost) 
    {
      improved = true;
      impl_->current_solution.cost = current.g_cost;
      impl_->current_solution.epsilon = impl_->epsilon;
      impl_->current_solution.path = reconstructPath(impl_->goal_node);
      break;
    }
    
    current.setOpen(false);
    current.setClosed(true);
    impl_->stats.nodes_expanded++;
    
    expandNode(current_id);
  }
  
  impl_->stats.current_epsilon = impl_->epsilon;
  impl_->stats.solution_cost = impl_->current_solution.cost;
  
  return improved;
}

PlanResult ARAStarSearch::getCurrentSolution() const 
{
  return impl_->current_solution;
}

ARAStarSearch::Stats ARAStarSearch::getStats() const 
{
  return impl_->stats;
}

void ARAStarSearch::setHeuristicFunction(size_t index, HeuristicFunction func) 
{
  if (index < impl_->heuristic_functions.size()) 
  {
    impl_->heuristic_functions[index] = func;
  }
}

void ARAStarSearch::setCostFunction(CostFunction func) 
{
  impl_->cost_function = func;
}

void ARAStarSearch::setValidityFunction(ValidityFunction func) 
{
  impl_->validity_function = func;
}

void ARAStarSearch::expandNode(NodeId node_id) 
{
  auto& current = impl_->getNode(node_id, impl_->current_level);
  
  auto neighbors = impl_->grid->getNeighbors(node_id, impl_->current_level);
  
  for (NodeId neighbor_id : neighbors) 
  {
    Eigen::Vector3f neighbor_pos = impl_->grid->getNodePosition(neighbor_id, impl_->current_level);
    
    if (!impl_->isValid(neighbor_pos)) 
    {
      continue;
    }
    
    float edge_cost = impl_->getEdgeCost(node_id, neighbor_id);
    float tentative_g = current.g_cost + edge_cost;
    
    auto& neighbor = impl_->getNode(neighbor_id, impl_->current_level);
    
    if (tentative_g < neighbor.g_cost) 
    {
      neighbor.g_cost = tentative_g;
      neighbor.back_pointer = node_id;
      
      if (neighbor.h_cost == 0.0f) 
      {
        neighbor.h_cost = impl_->computeHeuristic(neighbor_pos);
      }
      
      if (neighbor.isClosed()) 
      {
        impl_->level_states[static_cast<size_t>(impl_->current_level)].inconsistent_set.insert(neighbor_id);
      }
      else 
      {
        impl_->insertOpen(neighbor_id);
      }
    }
  }
}

void ARAStarSearch::updateNode(NodeId node_id, NodeId parent_id, float new_cost) 
{
  auto& node = impl_->getNode(node_id, impl_->current_level);
  
  if (new_cost < node.g_cost) 
  {
    node.g_cost = new_cost;
    node.back_pointer = parent_id;
    
    if (node.isClosed()) 
    {
      impl_->level_states[static_cast<size_t>(impl_->current_level)].inconsistent_set.insert(node_id);
    }
    else 
    {
      impl_->updateKey(node_id);
    }
  }
}

std::vector<Eigen::Vector3f> ARAStarSearch::reconstructPath(NodeId goal_id) const 
{
  std::vector<Eigen::Vector3f> path;
  
  NodeId current_id = goal_id;
  while (current_id != INVALID_NODE) 
  {
    path.push_back(impl_->grid->getNodePosition(current_id, impl_->current_level));
    
    auto& level_state = impl_->level_states[static_cast<size_t>(impl_->current_level)];
    auto it = level_state.nodes.find(current_id);
    if (it == level_state.nodes.end() || it->second.back_pointer == INVALID_NODE) 
    {
      break;
    }
    
    current_id = it->second.back_pointer;
  }
  
  std::reverse(path.begin(), path.end());
  return path;
}

SearchNode::HeuristicSet ARAStarSearch::computeHeuristics(const Eigen::Vector3f& pos) const 
{
  SearchNode::HeuristicSet h;
  
  h.h0 = (pos - impl_->goal_position).norm();
  
  if (impl_->heuristic_functions[1]) 
  {
    h.h1 = impl_->heuristic_functions[1](pos, impl_->goal_position);
  }
  
  if (impl_->heuristic_functions[2]) 
  {
    h.h2 = impl_->heuristic_functions[2](pos, impl_->goal_position);
  }
  
  return h;
}

void ARAStarSearch::batchComputeHeuristics(const Eigen::Vector3f* positions, size_t count,
                                          SearchNode::HeuristicSet* results) const 
{
  tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
    [&](const tbb::blocked_range<size_t>& range) 
    {
      for (size_t i = range.begin(); i != range.end(); ++i) 
      {
        results[i] = computeHeuristics(positions[i]);
      }
    });
}

void ARAStarSearch::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["parameters"]["epsilon"] = impl_->epsilon;
  j["parameters"]["weights"] = 
  {
    impl_->w0, impl_->w1, impl_->w2
  };
  j["parameters"]["current_level"] = static_cast<int>(impl_->current_level);
  
  j["stats"]["nodes_expanded"] = impl_->stats.nodes_expanded;
  j["stats"]["nodes_generated"] = impl_->stats.nodes_generated;
  j["stats"]["open_list_size"] = impl_->stats.open_list_size;
  j["stats"]["solution_cost"] = impl_->stats.solution_cost;
  
  if (impl_->current_solution.isSuccess()) 
  {
    j["solution"]["cost"] = impl_->current_solution.cost;
    j["solution"]["epsilon"] = impl_->current_solution.epsilon;
    j["solution"]["path_length"] = impl_->current_solution.path_length;
    j["solution"]["waypoints"] = impl_->current_solution.path.size();
  }
  
  for (size_t i = 0; i < impl_->level_states.size(); ++i) 
  {
    const auto& level_state = impl_->level_states[i];
    auto& level_json = j["levels"][i];
    
    level_json["nodes_total"] = level_state.nodes.size();
    level_json["nodes_closed"] = level_state.closed_set.size();
    level_json["nodes_inconsistent"] = level_state.inconsistent_set.size();
  }
  
  std::ofstream ofs(filename);
  ofs << j.dump(2);
}

}
}
}