#ifndef VKR_PLANNING_AMCP_PLANNER_HPP_
#define VKR_PLANNING_AMCP_PLANNER_HPP_

#include "vkr/planning/plan_request.hpp"
#include "vkr/planning/plan_result.hpp"
#include "vkr/planning/search/ara_star.hpp"
#include "vkr/planning/search/dstar_lite.hpp"
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <future>
#include <tbb/concurrent_queue.h>

namespace vkr
{

namespace terrain
{
  class WaveletGrid;
}

namespace geometry
{
namespace collision
{
  class CollisionQuery;
}
namespace corridors
{
  class SpatialCorridorMap;
  struct CorridorFreeEvent;
}
}

namespace planning
{

class AMCPlanner
{
public:
  explicit AMCPlanner(const VkrConfig& config);
  ~AMCPlanner();
  
  void setTerrain(std::shared_ptr<terrain::WaveletGrid> terrain);
  void setCollisionQuery(std::shared_ptr<geometry::collision::CollisionQuery> collision);
  void setCorridors(std::shared_ptr<geometry::corridors::SpatialCorridorMap> corridors);
  
  PlanResult plan(const PlanRequest& request);
  
  std::future<PlanResult> planAsync(const PlanRequest& request);
  
  bool cancelAsync();
  
  void pushEvent(const geometry::corridors::CorridorFreeEvent& event);
  void updateEdgeCost(NodeId from, NodeId to, float new_cost);
  
  void clear();
  
  struct Stats
  {
    size_t total_plans;
    size_t successful_plans;
    float average_planning_time_ms;
    float average_path_length;
    size_t current_open_list_size;
    float current_epsilon;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  void processEvents();
  
  float computeEdgeCost(const Eigen::Vector3f& from,
                       const Eigen::Vector3f& to,
                       UAVId uav_id) const;
  
  float computeHeightPenalty(const Eigen::Vector3f& pos) const;
  float computeCorridorPenalty(const Eigen::Vector3f& pos, UAVId uav_id) const;
  
  bool isValidPosition(const Eigen::Vector3f& pos) const;
  bool isValidEdge(const Eigen::Vector3f& from, const Eigen::Vector3f& to) const;
};

}
}

#endif