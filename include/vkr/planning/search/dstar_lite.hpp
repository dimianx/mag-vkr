#ifndef VKR_PLANNING_SEARCH_DSTAR_LITE_HPP_
#define VKR_PLANNING_SEARCH_DSTAR_LITE_HPP_

#include "vkr/planning/search/search_node.hpp"
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <vector>
#include <tbb/concurrent_queue.h>

namespace vkr
{
namespace planning
{
namespace search
{

class DStarLite
{
public:
  explicit DStarLite(const PlanningConfig& config);
  ~DStarLite();
  
  void initialize(NodeId start, NodeId goal,
                 std::shared_ptr<MultiResolutionGrid> grid);
  
  struct EdgeUpdate
  {
    NodeId from;
    NodeId to;
    float new_cost;
    geometry::BoundingBox affected_region;
  };
  
  void updateEdgeCost(const EdgeUpdate& update);
  void batchUpdateEdgeCosts(const std::vector<EdgeUpdate>& updates);
  
  bool replan();
  
  std::vector<NodeId> getCurrentPath() const;
  
  bool needsReplan() const;
  
  void updateStart(NodeId new_start);
  
  struct Stats
  {
    size_t nodes_updated;
    size_t replan_iterations;
    float replan_time_ms;
    size_t queue_size;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  void updateNode(NodeId node);
  void computeShortestPath();
  SearchNode::Key calculateKey(NodeId node) const;
  
  void propagateUpdate(NodeId node);
  std::vector<NodeId> getAffectedNodes(const geometry::BoundingBox& region) const;
  
  void batchUpdateNodes(const NodeId* nodes, size_t count);
};

}
}
}

#endif