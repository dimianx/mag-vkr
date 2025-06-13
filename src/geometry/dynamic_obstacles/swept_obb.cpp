#include "vkr/geometry/dynamic_obstacles/swept_obb.hpp"
#include "vkr/geometry/dynamic_obstacles/sob_builder.hpp"
#include "vkr/geometry/intersections.hpp"
#include <cmath>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

std::vector<Eigen::Vector3f> OBB::getCorners() const 
{
  std::vector<Eigen::Vector3f> corners(8);
  
  for (int i = 0; i < 8; ++i)
  {
    Eigen::Vector3f local_corner(
      (i & 1) ? half_extents.x() : -half_extents.x(),
      (i & 2) ? half_extents.y() : -half_extents.y(),
      (i & 4) ? half_extents.z() : -half_extents.z()
    );
    
    corners[i] = center + orientation * local_corner;
  }
  
  return corners;
}

BoundingBox OBB::getAABB() const 
{
  BoundingBox aabb;
  auto corners = getCorners();
  
  aabb.min = corners[0];
  aabb.max = corners[0];
  
  for (size_t i = 1; i < corners.size(); ++i)
  {
    aabb.expand(corners[i]);
  }
  
  return aabb;
}

OBB OBB::transform(const Eigen::Affine3f& transform) const 
{
  OBB result;
  result.center = transform * center;
  result.orientation = transform.rotation() * orientation;
  result.half_extents = half_extents;
  return result;
}

OBB SweptOBB::getOBBAtTime(const std::chrono::steady_clock::time_point& t) const
{
  float s = 0.0f;
  
  if (!isRepeating())
  {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    if (duration > 0)
    {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count();
      s = std::clamp(static_cast<float>(elapsed) / duration, 0.0f, 1.0f);
    }
  }
  else
  {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t - t0).count() / 1000.0f;
    
    float mapped_time = 0.0f;
    float t_power = 1.0f;
    for (float coeff : time_coeffs)
    {
      mapped_time += coeff * t_power;
      t_power *= elapsed;
    }
    
    s = mapped_time - std::floor(mapped_time);
  }
  
  Eigen::Vector3f position;
  
  if (position_coeffs.size() == 2)
  {
    position = (1.0f - s) * position_coeffs[0] + s * position_coeffs[1];
  }
  else
  {
    position = Eigen::Vector3f::Zero();
    float s_power = 1.0f;
    for (const auto& coeff : position_coeffs)
    {
      position += coeff * s_power;
      s_power *= s;
    }
  }
  
  Eigen::Vector3f euler_angles = Eigen::Vector3f::Zero();
  if (!rotation_coeffs.empty())
  {
    if (rotation_coeffs.size() == 2)
    {
      euler_angles = (1.0f - s) * rotation_coeffs[0] + s * rotation_coeffs[1];
    }
    else
    {
      float s_power_rot = 1.0f;
      for (const auto& coeff : rotation_coeffs)
      {
        euler_angles += coeff * s_power_rot;
        s_power_rot *= s;
      }
    }
  }
  
  Eigen::Matrix3f rotation = 
    Eigen::AngleAxisf(euler_angles.z(), Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(euler_angles.y(), Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(euler_angles.x(), Eigen::Vector3f::UnitX()).toRotationMatrix();
  
  OBB result;
  result.center = position;
  result.orientation = rotation * base_obb.orientation;
  result.half_extents = base_obb.half_extents;
  
  return result;
}

float SweptOBB::distanceToPoint(const Eigen::Vector3f& point, 
                                const std::chrono::steady_clock::time_point& t) const 
{
  OBB obb = getOBBAtTime(t);
  
  Eigen::Vector3f local_point = obb.orientation.transpose() * (point - obb.center);
  
  Eigen::Vector3f closest_local = local_point.cwiseMax(-obb.half_extents).cwiseMin(obb.half_extents);
  float distance = (local_point - closest_local).norm();
  
  if (intersectPointOBB(point, obb))
  {
    return -distance;
  }
  
  return distance;
}

float SweptOBB::distanceToOBB(const OBB& other, 
                              const std::chrono::steady_clock::time_point& t) const 
{
  OBB obb = getOBBAtTime(t);
  
  if (intersectOBBOBB(obb, other))
  {
    return 0.0f;
  }
  
  return (obb.center - other.center).norm() - 
         (obb.half_extents.norm() + other.half_extents.norm());
}

void SweptOBB::computeBounds() 
{
  const int num_samples = 20;
  
  bool first = true;
  
  for (int i = 0; i <= num_samples; ++i)
  {
    float s = static_cast<float>(i) / num_samples;
    
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    float s_power = 1.0f;
    for (const auto& coeff : position_coeffs)
    {
      position += coeff * s_power;
      s_power *= s;
    }
    
    Eigen::Vector3f euler_angles = Eigen::Vector3f::Zero();
    if (!rotation_coeffs.empty())
    {
      float s_power_rot = 1.0f;
      for (const auto& coeff : rotation_coeffs)
      {
        euler_angles += coeff * s_power_rot;
        s_power_rot *= s;
      }
    }
    
    Eigen::Matrix3f rotation = 
      Eigen::AngleAxisf(euler_angles.z(), Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(euler_angles.y(), Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(euler_angles.x(), Eigen::Vector3f::UnitX()).toRotationMatrix();
    
    OBB current_obb;
    current_obb.center = base_obb.center + position;
    current_obb.orientation = rotation * base_obb.orientation;
    current_obb.half_extents = base_obb.half_extents;
    
    BoundingBox current_aabb = current_obb.getAABB();
    
    if (first)
    {
      cached.spatial_aabb = current_aabb;
      first = false;
    }
    else
    {
      cached.spatial_aabb.expand(current_aabb.min);
      cached.spatial_aabb.expand(current_aabb.max);
    }
  }
  
  const int fine_samples = 50;
  std::vector<Eigen::Vector3f> sample_points;
  sample_points.reserve(fine_samples * 8);
  
  for (int i = 0; i <= fine_samples; ++i)
  {
    float s = static_cast<float>(i) / fine_samples;
    
    auto current_obb = getOBBAtTime(t0 + std::chrono::milliseconds(
        static_cast<long>(s * std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count())));
    
    auto corners = current_obb.getCorners();
    sample_points.insert(sample_points.end(), corners.begin(), corners.end());
  }
  
  cached.bounding_obb = computeOptimalOBB(sample_points);
}

Sphere SweptOBB::getTimeBoundingSphere() const 
{
  Sphere result;
  result.center = cached.spatial_aabb.center();
  
  float spatial_radius = (cached.spatial_aabb.max - cached.spatial_aabb.min).norm() * 0.5f;
  
  float time_radius = 1.0f;
  if (!isRepeating())
  {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    time_radius = duration / 1000.0f;
  }
  
  result.radius = std::sqrt(spatial_radius * spatial_radius + time_radius * time_radius);
  
  return result;
}

}
}
}