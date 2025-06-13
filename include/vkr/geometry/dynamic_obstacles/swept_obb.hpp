#ifndef VKR_GEOMETRY_DYNAMIC_OBSTACLES_SWEPT_OBB_HPP_
#define VKR_GEOMETRY_DYNAMIC_OBSTACLES_SWEPT_OBB_HPP_

#include "vkr/geometry/primitives.hpp"
#include "vkr/common.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <vector>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

struct OBB
{
  Eigen::Vector3f center;
  Eigen::Vector3f half_extents;
  Eigen::Matrix3f orientation;
  
  OBB() : center(Eigen::Vector3f::Zero()),
          half_extents(Eigen::Vector3f::Ones() * 0.1f),
          orientation(Eigen::Matrix3f::Identity()) {}
  
  std::vector<Eigen::Vector3f> getCorners() const;
  
  BoundingBox getAABB() const;
  
  OBB transform(const Eigen::Affine3f& transform) const;
};

struct SweptOBB
{
  OBB base_obb;
  
  std::vector<Eigen::Vector3f> position_coeffs;
  std::vector<Eigen::Vector3f> rotation_coeffs;
  std::vector<float> time_coeffs;
  
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t1 = t0 + std::chrono::seconds(1);
  
  ObjectId object_id = 0;
  
  struct CachedBounds
  {
    BoundingBox spatial_aabb = {Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones()};
    OBB bounding_obb;
  } cached;
  
  SweptOBB() = default;
  
  OBB getOBBAtTime(const std::chrono::steady_clock::time_point& t) const;
  
  bool isActiveAtTime(const std::chrono::steady_clock::time_point& t) const
  {
    return t >= t0 && t <= t1;
  }
  
  bool isRepeating() const
  {
    return !time_coeffs.empty();
  }
  
  float distanceToPoint(const Eigen::Vector3f& point,
                       const std::chrono::steady_clock::time_point& t) const;
  
  float distanceToOBB(const OBB& other,
                     const std::chrono::steady_clock::time_point& t) const;
  
  void computeBounds();
  
  Sphere getTimeBoundingSphere() const;
};

}
}
}

#endif