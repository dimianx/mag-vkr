#ifndef VKR_GEOMETRY_STATIC_OBSTACLES_BST_NODE_HPP_
#define VKR_GEOMETRY_STATIC_OBSTACLES_BST_NODE_HPP_

#include "vkr/geometry/primitives.hpp"
#include "vkr/common.hpp"
#include <vector>

namespace vkr
{
namespace geometry
{
namespace static_obstacles
{

struct BSTNode
{
  Sphere sphere;
  
  uint32_t left_child = INVALID_ID;
  uint32_t right_child = INVALID_ID;
  
  std::vector<uint32_t> face_indices;
  
  bool isLeaf() const
  {
    return left_child == INVALID_ID && right_child == INVALID_ID;
  }
  
  size_t getFaceCount() const
  {
    return face_indices.size();
  }
};

}
}
}

#endif