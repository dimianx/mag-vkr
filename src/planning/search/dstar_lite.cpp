#include "vkr/planning/search/dstar_lite.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <fstream>
#include <queue>
#include <algorithm>

namespace vkr
{
namespace planning
{
namespace search
{

struct DStarLite::Implementation 
{
  PlanningConfig config;
  std::shared_ptr<MultiResolutionGrid> grid;
  
  std::unordered_map<NodeId, SearchNode> nodes;
  
  struct OpenNode 
  {
    SearchNode::Key key;
    NodeId id;
    uint64_t heap_index = std::numeric_limits<uint64_t>::max();
  };
  
  std::vector<OpenNode> open_heap;
  std::unordered_map<NodeId, size_t> node_to_heap_index;
  
  NodeId start_node = INVALID_NODE;
  NodeId goal_node = INVALID_NODE;
  NodeId last_start = INVALID_NODE;
  
  float km = 0.0f;
  
  Stats stats
  {
  };
  
  tbb::concurrent_queue<EdgeUpdate> pending_updates;
  
  struct PathCache 
  {
    std::vector<NodeId> path;
    std::unordered_map<NodeId, size_t> node_to_path_index;
    bool valid = false;
    
    void rebuild(const std::vector<NodeId>& new_path) 
    {
      path = new_path;
      node_to_path_index.clear();
      
      for (size_t i = 0; i < path.size(); ++i) 
      {
        node_to_path_index[path[i]] = i;
      }
      valid = true;
    }
    
    bool contains(NodeId node) const 
    {
      return node_to_path_index.find(node) != node_to_path_index.end();
    }
    
    void invalidate() 
    {
      valid = false;
    }
  } path_cache;
  
  void pushOpen(NodeId id, const SearchNode::Key& key) 
  {
    OpenNode node
    {
      key, id, open_heap.size()
    };
    open_heap.push_back(node);
    node_to_heap_index[id] = node.heap_index;
    bubbleUp(node.heap_index);
  }
  
  bool popOpen(NodeId& id, SearchNode::Key& key) 
  {
    if (open_heap.empty()) return false;
    
    id = open_heap[0].id;
    key = open_heap[0].key;
    
    node_to_heap_index.erase(id);
    
    if (open_heap.size() > 1) 
    {
      open_heap[0] = open_heap.back();
      open_heap[0].heap_index = 0;
      node_to_heap_index[open_heap[0].id] = 0;
      open_heap.pop_back();
      bubbleDown(0);
    }
    else 
    {
      open_heap.clear();
    }
    
    return true;
  }
  
  void updateOpen(NodeId id, const SearchNode::Key& new_key) 
  {
    auto it = node_to_heap_index.find(id);
    if (it != node_to_heap_index.end()) 
    {
      size_t index = it->second;
      SearchNode::Key old_key = open_heap[index].key;
      open_heap[index].key = new_key;
      
      if (new_key < old_key) 
      {
        bubbleUp(index);
      }
      else 
      {
        bubbleDown(index);
      }
    }
    else 
    {
      pushOpen(id, new_key);
    }
  }
  
  void bubbleUp(size_t index) 
  {
    while (index > 0) 
    {
      size_t parent = (index - 1) / 2;
      if (open_heap[index].key < open_heap[parent].key) 
      {
        std::swap(open_heap[index], open_heap[parent]);
        open_heap[index].heap_index = index;
        open_heap[parent].heap_index = parent;
        node_to_heap_index[open_heap[index].id] = index;
        node_to_heap_index[open_heap[parent].id] = parent;
        index = parent;
      }
      else 
      {
        break;
      }
    }
  }
  
  void bubbleDown(size_t index) 
  {
    while (true) 
    {
      size_t smallest = index;
      size_t left = 2 * index + 1;
      size_t right = 2 * index + 2;
      
      if (left < open_heap.size() && open_heap[left].key < open_heap[smallest].key) 
      {
        smallest = left;
      }
      
      if (right < open_heap.size() && open_heap[right].key < open_heap[smallest].key) 
      {
        smallest = right;
      }
      
      if (smallest != index) 
      {
        std::swap(open_heap[index], open_heap[smallest]);
        open_heap[index].heap_index = index;
        open_heap[smallest].heap_index = smallest;
        node_to_heap_index[open_heap[index].id] = index;
        node_to_heap_index[open_heap[smallest].id] = smallest;
        index = smallest;
      }
      else 
      {
        break;
      }
    }
  }
  
  SearchNode& getNode(NodeId id) 
  {
    auto it = nodes.find(id);
    if (it == nodes.end()) 
    {
      SearchNode node;
      node.id = id;
      node.g_cost = INFINITY_F;
      node.rhs = INFINITY_F;
      nodes[id] = node;
      return nodes[id];
    }
    return it->second;
  }
  
  float heuristic(NodeId from, NodeId to) 
  {
    Eigen::Vector3f pos_from = grid->getNodePosition(from, MultiResolutionGrid::Level::FINE);
    Eigen::Vector3f pos_to = grid->getNodePosition(to, MultiResolutionGrid::Level::FINE);
    return (pos_to - pos_from).norm();
  }
  
  SearchNode::Key calculateKey(NodeId id) 
  {
    auto& node = getNode(id);
    float min_g_rhs = std::min(node.g_cost, node.rhs);
    return 
    {
      min_g_rhs + heuristic(start_node, id) + km,
      min_g_rhs
    };
  }
  
  void removeOpen(NodeId id)
  {
    auto it = node_to_heap_index.find(id);
    if (it == node_to_heap_index.end()) 
    {
      return;
    }
    
    size_t index_to_remove = it->second;
    size_t last_index = open_heap.size() - 1;
    
    if (index_to_remove != last_index) 
    {
      std::swap(open_heap[index_to_remove], open_heap[last_index]);
      
      open_heap[index_to_remove].heap_index = index_to_remove;
      node_to_heap_index[open_heap[index_to_remove].id] = index_to_remove;
    }
    
    open_heap.pop_back();
    node_to_heap_index.erase(id);
    
    if (index_to_remove != last_index && index_to_remove < open_heap.size()) 
    {
      bubbleUp(index_to_remove);
      bubbleDown(index_to_remove);
    }
  }

  void updateNode(NodeId id) 
  {
    if (id == INVALID_NODE) 
    {
      return;
    }
    
    auto& node = getNode(id);
    
    if (id != goal_node) 
    {
      float min_rhs = INFINITY_F;
      auto neighbors = grid->getNeighbors(id, MultiResolutionGrid::Level::FINE);
      
      for (NodeId neighbor_id : neighbors) 
      {
        if (neighbor_id != INVALID_NODE) 
        {
          float edge_cost = grid->getEdgeCost(neighbor_id, id, MultiResolutionGrid::Level::FINE);
          if (edge_cost < INFINITY_F) 
          {
            auto& neighbor = getNode(neighbor_id);
            float cost = neighbor.g_cost + edge_cost;
            if (cost < min_rhs) 
            {
              min_rhs = cost;
            }
          }
        }
      }
      
      node.rhs = min_rhs;
    }
    
    removeOpen(id);
    
    if (node.g_cost != node.rhs) 
    {
      node.flags |= SearchNode::FLAG_INCONSISTENT;
      updateOpen(id, calculateKey(id));
      stats.queue_size = open_heap.size();
    }
    else 
    {
      node.flags &= ~SearchNode::FLAG_INCONSISTENT;
    }
    
    if (path_cache.contains(id)) 
    {
      path_cache.invalidate();
    }
  }
    
  void reset() 
  {
    nodes.clear();
    open_heap.clear();
    node_to_heap_index.clear();
    km = 0.0f;
    stats = Stats
    {
    };
    path_cache.invalidate();
  }
};

DStarLite::DStarLite(const PlanningConfig& config)
  : impl_(std::make_unique<Implementation>())
{
  impl_->config = config;
}

DStarLite::~DStarLite() = default;

void DStarLite::initialize(NodeId start, NodeId goal, 
                          std::shared_ptr<MultiResolutionGrid> grid) 
{
  impl_->reset();
  
  impl_->grid = grid;
  impl_->start_node = start;
  impl_->goal_node = goal;
  impl_->last_start = start;
  
  auto& goal_node = impl_->getNode(goal);
  goal_node.rhs = 0.0f;
  impl_->updateNode(goal);
  
  computeShortestPath();
  
  auto path = getCurrentPath();
  impl_->path_cache.rebuild(path);
}

void DStarLite::updateEdgeCost(const EdgeUpdate& update) 
{
  impl_->grid->updateEdgeCost(update.from, update.to, update.new_cost, 
                             MultiResolutionGrid::Level::FINE);
  
  impl_->updateNode(update.from);
  impl_->updateNode(update.to);
  
  impl_->path_cache.invalidate();
  
  impl_->stats.nodes_updated += 2;
}

void DStarLite::batchUpdateEdgeCosts(const std::vector<EdgeUpdate>& updates) 
{
  tbb::parallel_for(tbb::blocked_range<size_t>(0, updates.size()),
    [&](const tbb::blocked_range<size_t>& range) 
    {
      for (size_t i = range.begin(); i != range.end(); ++i) 
      {
        impl_->grid->updateEdgeCost(updates[i].from, updates[i].to, 
                                   updates[i].new_cost, MultiResolutionGrid::Level::FINE);
      }
    });
  
  std::unordered_set<NodeId> affected_nodes;
  for (const auto& update : updates) 
  {
    affected_nodes.insert(update.from);
    affected_nodes.insert(update.to);
    
    if (update.affected_region.min.allFinite() && update.affected_region.max.allFinite()) 
    {
      auto region_nodes = getAffectedNodes(update.affected_region);
      affected_nodes.insert(region_nodes.begin(), region_nodes.end());
    }
  }
  
  for (NodeId id : affected_nodes) 
  {
    impl_->updateNode(id);
  }
  
  impl_->path_cache.invalidate();
  impl_->stats.nodes_updated += affected_nodes.size();
}

bool DStarLite::replan() 
{
  auto start_time = std::chrono::steady_clock::now();
  
  computeShortestPath();
  
  auto path = getCurrentPath();
  bool success = !path.empty();
  
  auto elapsed = std::chrono::steady_clock::now() - start_time;
  impl_->stats.replan_time_ms = std::chrono::duration<float, std::milli>(elapsed).count();
  impl_->stats.replan_iterations++;
  
  return success;
}

std::vector<NodeId> DStarLite::getCurrentPath() const 
{
  if (impl_->path_cache.valid) 
  {
    return impl_->path_cache.path;
  }
  
  std::vector<NodeId> path;
  
  if (impl_->start_node == INVALID_NODE || impl_->goal_node == INVALID_NODE) 
  {
    return path;
  }
  
  NodeId current = impl_->start_node;
  path.push_back(current);
  
  std::unordered_set<NodeId> visited;
  visited.insert(current);
  
  while (current != impl_->goal_node && path.size() < 10000) 
  {
    auto neighbors = impl_->grid->getNeighbors(current, MultiResolutionGrid::Level::FINE);
    
    NodeId best_neighbor = INVALID_NODE;
    float best_cost = INFINITY_F;
    
    for (NodeId neighbor_id : neighbors) 
    {
      if (visited.count(neighbor_id) > 0) 
      {
        continue;
      }
      
      auto& neighbor = impl_->getNode(neighbor_id);
      float edge_cost = impl_->grid->getEdgeCost(current, neighbor_id, 
                                                 MultiResolutionGrid::Level::FINE);
      float total_cost = neighbor.g_cost + edge_cost;
      
      if (total_cost < best_cost) 
      {
        best_cost = total_cost;
        best_neighbor = neighbor_id;
      }
    }
    
    if (best_neighbor == INVALID_NODE) 
    {
      return 
      {
      };
    }
    
    current = best_neighbor;
    path.push_back(current);
    visited.insert(current);
  }
  
  impl_->path_cache.rebuild(path);
  
  return path;
}

bool DStarLite::needsReplan() const 
{
  return !impl_->path_cache.valid || !impl_->open_heap.empty();
}

void DStarLite::updateStart(NodeId new_start) 
{
  if (new_start == impl_->start_node) 
  {
    return;
  }
  
  impl_->km += impl_->heuristic(impl_->last_start, new_start);
  impl_->last_start = impl_->start_node;
  impl_->start_node = new_start;
  
  impl_->path_cache.invalidate();
  
  computeShortestPath();
}

DStarLite::Stats DStarLite::getStats() const 
{
  return impl_->stats;
}

void DStarLite::computeShortestPath() 
{
  auto deadline = std::chrono::steady_clock::now() + 
                  std::chrono::milliseconds(static_cast<int>(impl_->config.max_runtime_ms));
  
  int iterations = 0;
  const int MAX_ITERATIONS = 500000;
  
  while (!impl_->open_heap.empty() && 
         std::chrono::steady_clock::now() < deadline &&
         iterations++ < MAX_ITERATIONS) 
  {
    NodeId current_id;
    SearchNode::Key key;
    
    if (!impl_->popOpen(current_id, key)) 
    {
      break;
    }
    
    if (current_id == INVALID_NODE || current_id == impl_->start_node) 
    {
      continue;
    }
    
    auto& start_node = impl_->getNode(impl_->start_node);
    SearchNode::Key start_key = impl_->calculateKey(impl_->start_node);
    
    if (key >= start_key && start_node.rhs == start_node.g_cost) 
    {
      break;
    }
    
    auto& current = impl_->getNode(current_id);
    SearchNode::Key current_key = impl_->calculateKey(current_id);
    
    if (current_key < key) 
    {
      impl_->updateOpen(current_id, current_key);
      continue;
    }
    
    current.setOpen(false);
    
    if (current.g_cost > current.rhs) 
    {
      current.g_cost = current.rhs;
      
      auto neighbors = impl_->grid->getNeighbors(current_id, MultiResolutionGrid::Level::FINE);
      for (NodeId neighbor_id : neighbors) 
      {
        if (neighbor_id != INVALID_NODE && neighbor_id != impl_->goal_node) 
        {
          impl_->updateNode(neighbor_id);
        }
      }
    }
    else 
    {
      current.g_cost = INFINITY_F;
      
      std::vector<NodeId> nodes_to_update;
      nodes_to_update.push_back(current_id);
      
      auto neighbors = impl_->grid->getNeighbors(current_id, MultiResolutionGrid::Level::FINE);
      for (NodeId neighbor_id : neighbors) 
      {
        if (neighbor_id != INVALID_NODE && neighbor_id != impl_->goal_node) 
        {
          nodes_to_update.push_back(neighbor_id);
        }
      }
      
      for (NodeId node_id : nodes_to_update) 
      {
        if (node_id != INVALID_NODE) 
        {
          impl_->updateNode(node_id);
        }
      }
    }
  }
  
  impl_->updateNode(impl_->start_node);
}

SearchNode::Key DStarLite::calculateKey(NodeId node) const 
{
  return impl_->calculateKey(node);
}

void DStarLite::updateNode(NodeId node) 
{
  impl_->updateNode(node);
  impl_->stats.nodes_updated++;
}

void DStarLite::propagateUpdate(NodeId node) 
{
  auto neighbors = impl_->grid->getNeighbors(node, MultiResolutionGrid::Level::FINE);
  
  for (NodeId neighbor_id : neighbors) 
  {
    impl_->updateNode(neighbor_id);
  }
  
  impl_->stats.nodes_updated += neighbors.size();
}

std::vector<NodeId> DStarLite::getAffectedNodes(const geometry::BoundingBox& region) const 
{
  std::vector<NodeId> affected_nodes;
  
  const float resolution = impl_->grid->getResolution(MultiResolutionGrid::Level::FINE);
  
  Eigen::Vector3f min_pos = region.min;
  Eigen::Vector3f max_pos = region.max;
  
  for (float x = min_pos.x(); x <= max_pos.x(); x += resolution) 
  {
    for (float y = min_pos.y(); y <= max_pos.y(); y += resolution) 
    {
      for (float z = min_pos.z(); z <= max_pos.z(); z += resolution) 
      {
        Eigen::Vector3f pos(x, y, z);
        NodeId node_id = impl_->grid->getNode(pos, MultiResolutionGrid::Level::FINE);
        
        if (node_id != INVALID_NODE) 
        {
          affected_nodes.push_back(node_id);
        }
      }
    }
  }
  
  return affected_nodes;
}

void DStarLite::batchUpdateNodes(const NodeId* nodes, size_t count) 
{
  tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
    [&](const tbb::blocked_range<size_t>& range) 
    {
      for (size_t i = range.begin(); i != range.end(); ++i) 
      {
        impl_->updateNode(nodes[i]);
      }
    });
  
  impl_->stats.nodes_updated += count;
}

void DStarLite::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["state"]["start_node"] = impl_->start_node;
  j["state"]["goal_node"] = impl_->goal_node;
  j["state"]["km"] = impl_->km;
  
  j["stats"]["nodes_updated"] = impl_->stats.nodes_updated;
  j["stats"]["replan_iterations"] = impl_->stats.replan_iterations;
  j["stats"]["replan_time_ms"] = impl_->stats.replan_time_ms;
  j["stats"]["queue_size"] = impl_->stats.queue_size;
  
  j["path"]["valid"] = impl_->path_cache.valid;
  j["path"]["length"] = impl_->path_cache.valid ? impl_->path_cache.path.size() : 0;
  
  size_t consistent_count = 0;
  size_t inconsistent_count = 0;
  
  for (const auto& [id, node] : impl_->nodes) 
  {
    if (node.isInconsistent()) 
    {
      inconsistent_count++;
    }
    else if (node.g_cost < INFINITY_F) 
    {
      consistent_count++;
    }
  }
  
  j["nodes"]["total"] = impl_->nodes.size();
  j["nodes"]["consistent"] = consistent_count;
  j["nodes"]["inconsistent"] = inconsistent_count;
  
  j["open_list"]["size"] = impl_->open_heap.size();
  j["open_list"]["indexed_nodes"] = impl_->node_to_heap_index.size();
  
  std::ofstream ofs(filename);
  ofs << j.dump(2);
}

}
}
}