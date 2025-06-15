#ifndef VKR_GEOMETRY_COLLISION_HIT_RESULT_HPP_
#define VKR_GEOMETRY_COLLISION_HIT_RESULT_HPP_

#include "vkr/common.hpp"
#include <Eigen/Core>

namespace vkr
{
namespace geometry
{
namespace collision
{

struct HitResult
{
  bool hit = false;
  float t_min = INFINITY_F;
  ObjectId object_id = INVALID_ID;
  Eigen::Vector3f hit_point;
  Eigen::Vector3f hit_normal;
  
  enum class ObjectType
  {
    NONE = 0,
    STATIC_OBSTACLE,
    CORRIDOR,
    TERRAIN
  } object_type = ObjectType::NONE;
};

}
}
}

#endif