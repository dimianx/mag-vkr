#ifndef VKR_GEOMETRY_PRIMITIVES_HPP_
#define VKR_GEOMETRY_PRIMITIVES_HPP_

#include <Eigen/Core>

namespace vkr
{
namespace geometry
{

struct Sphere
{
  Eigen::Vector3f center;
  float radius;
};

struct Capsule
{
  Eigen::Vector3f p0;
  Eigen::Vector3f p1;
  float radius;
};

struct LineSegment
{
  Eigen::Vector3f start;
  Eigen::Vector3f end;
};

struct BoundingBox
{
  Eigen::Vector3f min;
  Eigen::Vector3f max;
  
  bool contains(const Eigen::Vector3f& point) const
  {
    return (point.array() >= min.array()).all() &&
           (point.array() <= max.array()).all();
  }
  
  bool intersects(const BoundingBox& other) const
  {
    return (min.array() <= other.max.array()).all() &&
           (max.array() >= other.min.array()).all();
  }
  
  void expand(const Eigen::Vector3f& point)
  {
    min = min.cwiseMin(point);
    max = max.cwiseMax(point);
  }
  
  void expand(const BoundingBox& other)
  {
    min = min.cwiseMin(other.min);
    max = max.cwiseMax(other.max);
  }
  
  Eigen::Vector3f center() const
  {
    return (min + max) * 0.5f;
  }
  
  Eigen::Vector3f size() const
  {
    return max - min;
  }
};

}
}

#endif