#ifndef VKR_GEOMETRY_INTERSECTIONS_HPP_
#define VKR_GEOMETRY_INTERSECTIONS_HPP_

#include "vkr/geometry/primitives.hpp"
#include "vkr/geometry/dynamic_obstacles/swept_obb.hpp"
#include "vkr/common.hpp"
#include <cmath>
#include <algorithm>

namespace vkr
{
namespace geometry
{

inline bool intersectSphereSphere(const Sphere& a, const Sphere& b) 
{
  float dist_sq = (a.center - b.center).squaredNorm();
  float radius_sum = a.radius + b.radius;
  return dist_sq <= radius_sum * radius_sum;
}

inline bool intersectSegmentSphere(const LineSegment& seg, const Sphere& sph) 
{
  Eigen::Vector3f d = seg.end - seg.start;
  Eigen::Vector3f f = seg.start - sph.center;
  
  float a = d.dot(d);
  float b = 2.0f * f.dot(d);
  float c = f.dot(f) - sph.radius * sph.radius;
  
  float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) 
  {
    return false;
  }
  
  discriminant = std::sqrt(discriminant);
  float t1 = (-b - discriminant) / (2.0f * a);
  float t2 = (-b + discriminant) / (2.0f * a);
  
  return (t1 >= 0.0f && t1 <= 1.0f) || (t2 >= 0.0f && t2 <= 1.0f) ||
         (t1 < 0.0f && t2 > 1.0f);
}

inline float segmentSegmentDistSq(const Eigen::Vector3f& p1,
                                  const Eigen::Vector3f& q1,
                                  const Eigen::Vector3f& p2,
                                  const Eigen::Vector3f& q2,
                                  float& sOut, float& tOut)
{
  const Eigen::Vector3f d1 = q1 - p1;
  const Eigen::Vector3f d2 = q2 - p2;
  const Eigen::Vector3f r  = p1 - p2;
  const float a = d1.dot(d1);
  const float e = d2.dot(d2);
  const float EPS = 1e-6f;

  float s = 0.0f, t = 0.0f;

  if (a < EPS && e < EPS)
  {
    sOut = tOut = 0.0f;
    return r.squaredNorm();
  }

  if (a < EPS)
  {
    s = 0.0f;
    t = std::clamp(d2.dot(r) / e, 0.0f, 1.0f);
  }
  else
  {
    const float c = d1.dot(r);
    if (e < EPS)
    {
      t = 0.0f;
      s = std::clamp(-c / a, 0.0f, 1.0f);
    }
    else
    {
      const float b     = d1.dot(d2);
      const float denom = a*e - b*b;

      if (denom > EPS)
        s = std::clamp((b*d2.dot(r) - c*e) / denom, 0.0f, 1.0f);
      else
        s = 0.0f;

      t = (b*s + d2.dot(r)) / e;

      if (t < 0.0f)
      {
        t = 0.0f;
        s = std::clamp(-c / a, 0.0f, 1.0f);
      }
      else if (t > 1.0f)
      {
        t = 1.0f;
        s = std::clamp((b - c) / a, 0.0f, 1.0f);
      }
    }
  }

  sOut = s;  tOut = t;
  Eigen::Vector3f c1 = p1 + d1 * s;
  Eigen::Vector3f c2 = p2 + d2 * t;
  return (c1 - c2).squaredNorm();
}

inline bool intersectCapsuleCapsule(const Capsule& a, const Capsule& b)
{
  const float rSum = a.radius + b.radius;
  float s, t;
  float dist2 = segmentSegmentDistSq(a.p0, a.p1, b.p0, b.p1, s, t);
  return dist2 <= rSum * rSum;
}

inline bool intersectSegmentCapsule(const LineSegment& seg, const Capsule& cap) 
{
  Capsule seg_capsule;
  seg_capsule.p0 = seg.start;
  seg_capsule.p1 = seg.end;
  seg_capsule.radius = 0.0f;
  
  return intersectCapsuleCapsule(seg_capsule, cap);
}

void batchIntersectSphereSphere(const Sphere* a, const Sphere* b, 
                                size_t count, bool* results);

void batchIntersectCapsuleCapsule(const Capsule* a, const Capsule* b, 
                                  size_t count, bool* results);

inline bool intersectOBBOBB(const dynamic_obstacles::OBB& a, const dynamic_obstacles::OBB& b) 
{
  Eigen::Matrix3f R = a.orientation.transpose() * b.orientation;
  
  Eigen::Vector3f t = a.orientation.transpose() * (b.center - a.center);
  
  Eigen::Matrix3f AbsR = R.cwiseAbs() + Eigen::Matrix3f::Constant(GEOM_EPS);
  
  for (int i = 0; i < 3; ++i) 
  {
    float ra = a.half_extents[i];
    float rb = b.half_extents.dot(AbsR.row(i));
    if (std::abs(t[i]) > ra + rb) return false;
  }
  
  for (int i = 0; i < 3; ++i) 
  {
    float ra = a.half_extents.dot(AbsR.col(i));
    float rb = b.half_extents[i];
    if (std::abs(t.dot(R.col(i))) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[1] * AbsR(2,0) + a.half_extents[2] * AbsR(1,0);
    float rb = b.half_extents[1] * AbsR(0,2) + b.half_extents[2] * AbsR(0,1);
    if (std::abs(t[2] * R(1,0) - t[1] * R(2,0)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[1] * AbsR(2,1) + a.half_extents[2] * AbsR(1,1);
    float rb = b.half_extents[0] * AbsR(0,2) + b.half_extents[2] * AbsR(0,0);
    if (std::abs(t[2] * R(1,1) - t[1] * R(2,1)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[1] * AbsR(2,2) + a.half_extents[2] * AbsR(1,2);
    float rb = b.half_extents[0] * AbsR(0,1) + b.half_extents[1] * AbsR(0,0);
    if (std::abs(t[2] * R(1,2) - t[1] * R(2,2)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(2,0) + a.half_extents[2] * AbsR(0,0);
    float rb = b.half_extents[1] * AbsR(1,2) + b.half_extents[2] * AbsR(1,1);
    if (std::abs(t[0] * R(2,0) - t[2] * R(0,0)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(2,1) + a.half_extents[2] * AbsR(0,1);
    float rb = b.half_extents[0] * AbsR(1,2) + b.half_extents[2] * AbsR(1,0);
    if (std::abs(t[0] * R(2,1) - t[2] * R(0,1)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(2,2) + a.half_extents[2] * AbsR(0,2);
    float rb = b.half_extents[0] * AbsR(1,1) + b.half_extents[1] * AbsR(1,0);
    if (std::abs(t[0] * R(2,2) - t[2] * R(0,2)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(1,0) + a.half_extents[1] * AbsR(0,0);
    float rb = b.half_extents[1] * AbsR(2,2) + b.half_extents[2] * AbsR(2,1);
    if (std::abs(t[1] * R(0,0) - t[0] * R(1,0)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(1,1) + a.half_extents[1] * AbsR(0,1);
    float rb = b.half_extents[0] * AbsR(2,2) + b.half_extents[2] * AbsR(2,0);
    if (std::abs(t[1] * R(0,1) - t[0] * R(1,1)) > ra + rb) return false;
  }
  
  {
    float ra = a.half_extents[0] * AbsR(1,2) + a.half_extents[1] * AbsR(0,2);
    float rb = b.half_extents[0] * AbsR(2,1) + b.half_extents[1] * AbsR(2,0);
    if (std::abs(t[1] * R(0,2) - t[0] * R(1,2)) > ra + rb) return false;
  }
  
  return true;
}

inline bool intersectPointOBB(const Eigen::Vector3f& point, const dynamic_obstacles::OBB& obb) 
{
  Eigen::Vector3f local_point = obb.orientation.transpose() * (point - obb.center);
  
  return std::abs(local_point.x()) <= obb.half_extents.x() &&
         std::abs(local_point.y()) <= obb.half_extents.y() &&
         std::abs(local_point.z()) <= obb.half_extents.z();
}

inline bool intersectSegmentOBB(const LineSegment& segment, const dynamic_obstacles::OBB& obb) 
{
  Eigen::Vector3f local_start = obb.orientation.transpose() * (segment.start - obb.center);
  Eigen::Vector3f local_end = obb.orientation.transpose() * (segment.end - obb.center);
  
  Eigen::Vector3f ray_dir = local_end - local_start;
  float ray_length = ray_dir.norm();
  
  if (ray_length < GEOM_EPS)
  {
    return intersectPointOBB(segment.start, obb);
  }
  
  ray_dir /= ray_length;
  
  float t_min = 0.0f;
  float t_max = ray_length;
  
  for (int i = 0; i < 3; ++i)
  {
    if (std::abs(ray_dir[i]) < GEOM_EPS)
    {
      if (std::abs(local_start[i]) > obb.half_extents[i])
      {
        return false;
      }
    }
    else
    {
      float t1 = (-obb.half_extents[i] - local_start[i]) / ray_dir[i];
      float t2 = (obb.half_extents[i] - local_start[i]) / ray_dir[i];
      
      if (t1 > t2) std::swap(t1, t2);
      
      t_min = std::max(t_min, t1);
      t_max = std::min(t_max, t2);
      
      if (t_min > t_max) return false;
    }
  }
  
  return true;
}

}
}

#endif