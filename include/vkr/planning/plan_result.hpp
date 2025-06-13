#ifndef VKR_PLANNING_PLAN_RESULT_HPP_
#define VKR_PLANNING_PLAN_RESULT_HPP_

#include "vkr/common.hpp"
#include <Eigen/Core>
#include <vector>
#include <chrono>

namespace vkr
{
namespace planning
{

struct PlanResult
{
  std::vector<Eigen::Vector3f> path;
  float cost = INFINITY_F;
  float epsilon = 1.0f;
  
  Status status = Status::FAILURE;
  
  std::chrono::milliseconds planning_time{0};
  size_t nodes_expanded = 0;
  size_t nodes_generated = 0;
  
  float path_length = 0.0f;
  float min_clearance = 0.0f;
  float max_altitude = 0.0f;
  
  bool isSuccess() const
  {
    return status == Status::SUCCESS && !path.empty();
  }
  
  size_t getWaypointCount() const
  {
    return path.size();
  }
};

}
}

#endif