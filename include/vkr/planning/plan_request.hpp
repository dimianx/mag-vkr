#ifndef VKR_PLANNING_PLAN_REQUEST_HPP_
#define VKR_PLANNING_PLAN_REQUEST_HPP_

#include "vkr/common.hpp"
#include "vkr/geometry/primitives.hpp"
#include <Eigen/Core>

namespace vkr
{
namespace planning
{

struct PlanRequest
{
  Eigen::Vector3f start;
  Eigen::Vector3f goal;
  
  CorridorId corridor_id = INVALID_ID;
  UAVId uav_id = INVALID_ID;
  
  float max_runtime_ms = 100.0f;
  
  geometry::BoundingBox world_bounds;
  bool has_world_bounds = false;
  
  bool isValid() const
  {
    return start.allFinite() && goal.allFinite() &&
           max_runtime_ms > 0.0f;
  }
  
  void setWorldBounds(const geometry::BoundingBox& bounds)
  {
    world_bounds = bounds;
    has_world_bounds = true;
  }
};

}
}

#endif