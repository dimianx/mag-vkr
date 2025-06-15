#ifndef VKR_GEOMETRY_INTERSECTIONS_HPP_
#define VKR_GEOMETRY_INTERSECTIONS_HPP_

#include "vkr/geometry/primitives.hpp"
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

}
}

#endif