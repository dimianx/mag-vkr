#ifndef VKR_PLANNING_SEARCH_ARA_STAR_HPP_
#define VKR_PLANNING_SEARCH_ARA_STAR_HPP_

#include "vkr/planning/search/search_node.hpp"
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/planning/plan_request.hpp"
#include "vkr/planning/plan_result.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <functional>
#include <tbb/concurrent_priority_queue.h>

namespace vkr
{
namespace planning
{
namespace search
{

class ARAStarSearch
{
public:
  explicit ARAStarSearch(const PlanningConfig& config);
  ~ARAStarSearch();
  
  void setGrid(std::shared_ptr<MultiResolutionGrid> grid);
  
  void setHeuristicWeights(float w0, float w1, float w2);
  
  PlanResult search(const PlanRequest& request);
  
  bool improveSolution(float new_epsilon, float time_limit_ms);
  
  PlanResult getCurrentSolution() const;
  MultiResolutionGrid::Level getCurrentSearchLevel() const;
  
  struct Stats
  {
    size_t nodes_expanded;
    size_t nodes_generated;
    size_t open_list_size;
    float current_epsilon;
    float solution_cost;
  };
  
  Stats getStats() const;
  
  using HeuristicFunction = std::function<float(const Eigen::Vector3f&, const Eigen::Vector3f&)>;
  using CostFunction = std::function<float(NodeId, NodeId)>;
  using ValidityFunction = std::function<bool(const Eigen::Vector3f&)>;
  
  void setHeuristicFunction(size_t index, HeuristicFunction func);
  void setCostFunction(CostFunction func);
  void setValidityFunction(ValidityFunction func);
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct NodeComparator
  {
    bool operator()(const std::pair<SearchNode::Key, NodeId>& a,
                   const std::pair<SearchNode::Key, NodeId>& b) const
    {
      return b.first < a.first;
    }
  };
  
  using OpenList = tbb::concurrent_priority_queue<
      std::pair<SearchNode::Key, NodeId>,
      NodeComparator>;
  
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  void expandNode(NodeId node_id);
  void updateNode(NodeId node_id, NodeId parent_id, float new_cost);
  std::vector<Eigen::Vector3f> reconstructPath(NodeId goal_id) const;
  
  SearchNode::HeuristicSet computeHeuristics(const Eigen::Vector3f& pos) const;
  
  void batchComputeHeuristics(const Eigen::Vector3f* positions, size_t count,
                             SearchNode::HeuristicSet* results) const;
};

}
}
}

#endif