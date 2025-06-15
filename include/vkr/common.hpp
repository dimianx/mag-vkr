#ifndef VKR_COMMON_HPP_
#define VKR_COMMON_HPP_

#include <cstdint>
#include <limits>

namespace vkr
{

using NodeId = uint32_t;
using CorridorId = uint32_t;
using SegmentId = uint32_t;
using UAVId = uint16_t;
using ObjectId = uint32_t;

constexpr float GEOM_EPS = 1e-6f;
constexpr float INFINITY_F = std::numeric_limits<float>::infinity();
constexpr uint32_t INVALID_ID = std::numeric_limits<uint32_t>::max();
constexpr NodeId INVALID_NODE = std::numeric_limits<NodeId>::max();

enum class Status
{
  SUCCESS = 0,
  FAILURE,
  TIMEOUT,
  IN_PROGRESS
};

}

#endif