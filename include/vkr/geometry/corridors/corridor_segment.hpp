#ifndef VKR_GEOMETRY_CORRIDORS_CORRIDOR_SEGMENT_HPP_
#define VKR_GEOMETRY_CORRIDORS_CORRIDOR_SEGMENT_HPP_

#include "vkr/geometry/primitives.hpp"
#include "vkr/common.hpp"
#include <atomic>
#include <chrono>

namespace vkr
{
namespace geometry
{
namespace corridors
{

struct CorridorSegment
{
  Capsule capsule;
  CorridorId corridor_id;
  SegmentId segment_id;
  std::atomic<uint32_t> next{INVALID_ID};
  
  BoundingBox getBoundingBox() const
  {
    BoundingBox bbox;
    bbox.min = capsule.p0.cwiseMin(capsule.p1) - Eigen::Vector3f::Constant(capsule.radius);
    bbox.max = capsule.p0.cwiseMax(capsule.p1) + Eigen::Vector3f::Constant(capsule.radius);
    return bbox;
  }
};

struct CorridorFreeEvent
{
  CorridorId corridor_id;
  SegmentId segment_id;
  std::chrono::steady_clock::time_point timestamp;
};

}
}
}

#endif