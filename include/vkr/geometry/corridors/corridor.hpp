#ifndef VKR_GEOMETRY_CORRIDORS_CORRIDOR_HPP_
#define VKR_GEOMETRY_CORRIDORS_CORRIDOR_HPP_

#include "vkr/common.hpp"
#include <vector>

namespace vkr
{
namespace geometry
{
namespace corridors
{

struct Corridor
{
  CorridorId id;
  UAVId owner;
  float radius;
  float coverage_fraction;
  size_t current_waypoint;
  size_t total_waypoints;
  std::vector<SegmentId> active_segments;
  
  bool isOwnedBy(UAVId uav_id) const
  {
    return owner == uav_id;
  }
  
  size_t getActiveSegmentCount() const
  {
    return active_segments.size();
  }
  
  float getRemainingPathFraction() const
  {
    if (total_waypoints <= 1) return 0.0f;
    size_t remaining = total_waypoints - current_waypoint - 1;
    return static_cast<float>(remaining) / static_cast<float>(total_waypoints - 1);
  }
  
  size_t getCoveredWaypoints() const
  {
    return active_segments.size() + current_waypoint;
  }
};

}
}
}

#endif