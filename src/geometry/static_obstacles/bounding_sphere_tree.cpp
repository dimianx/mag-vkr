#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/geometry/collision/hit_result.hpp"
#include "vkr/geometry/intersections.hpp"
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>
#include <spdlog/spdlog.h>
#include <unordered_set>

#include <Eigen/Geometry> 
#include <Eigen/Dense>     

namespace vkr
{
namespace geometry
{
namespace static_obstacles
{

BoundingSphereTree::BoundingSphereTree() = default;
BoundingSphereTree::~BoundingSphereTree() = default;

void BoundingSphereTree::build(const std::vector<Face>& faces, uint32_t max_faces_per_leaf)
{
  faces_ = faces;
  nodes_.clear();
  
  if (faces_.empty()) 
  {
    root_node_ = INVALID_ID;
    return;
  }
  
  nodes_.reserve(2 * faces_.size());
  
  std::vector<uint32_t> face_indices(faces_.size());
  std::iota(face_indices.begin(), face_indices.end(), 0);
  
  root_node_ = buildRecursive(face_indices, max_faces_per_leaf);
  
  spdlog::debug("Built BST with {} nodes for {} faces", nodes_.size(), faces_.size());
}

uint32_t BoundingSphereTree::buildRecursive(std::vector<uint32_t>& face_indices, 
                                           uint32_t max_faces_per_leaf)
{
  if (face_indices.empty()) 
  {
    return INVALID_ID;
  }
  
  uint32_t node_idx = static_cast<uint32_t>(nodes_.size());
  nodes_.emplace_back();
  
  Sphere sphere = computeBoundingSphere(face_indices);
  
  if (face_indices.size() <= max_faces_per_leaf) 
  {
    nodes_[node_idx].sphere = sphere;
    nodes_[node_idx].face_indices = std::move(face_indices);
    nodes_[node_idx].left_child = INVALID_ID;
    nodes_[node_idx].right_child = INVALID_ID;
    return node_idx;
  }
  
  Eigen::Vector3f centroid_sum = Eigen::Vector3f::Zero();
  for (uint32_t idx : face_indices) 
  {
    const Face& face = faces_[idx];
    Eigen::Vector3f face_center = (face.vertices[0] + face.vertices[1] + face.vertices[2]) / 3.0f;
    centroid_sum += face_center;
  }
  Eigen::Vector3f mean = centroid_sum / face_indices.size();
  
  Eigen::Vector3f variance = Eigen::Vector3f::Zero();
  for (uint32_t idx : face_indices) 
  {
    const Face& face = faces_[idx];
    Eigen::Vector3f face_center = (face.vertices[0] + face.vertices[1] + face.vertices[2]) / 3.0f;
    Eigen::Vector3f diff = face_center - mean;
    variance += diff.cwiseProduct(diff);
  }
  
  int split_axis = 0;
  variance.maxCoeff(&split_axis);
  
  std::sort(face_indices.begin(), face_indices.end(), 
    [this, split_axis](uint32_t a, uint32_t b) 
    {
      const Face& fa = faces_[a];
      const Face& fb = faces_[b];
      float ca = (fa.vertices[0][split_axis] + fa.vertices[1][split_axis] + fa.vertices[2][split_axis]) / 3.0f;
      float cb = (fb.vertices[0][split_axis] + fb.vertices[1][split_axis] + fb.vertices[2][split_axis]) / 3.0f;
      return ca < cb;
    });
  
  size_t mid = face_indices.size() / 2;
  std::vector<uint32_t> left_indices(face_indices.begin(), face_indices.begin() + mid);
  std::vector<uint32_t> right_indices(face_indices.begin() + mid, face_indices.end());
  
  uint32_t left_child = buildRecursive(left_indices, max_faces_per_leaf);
  uint32_t right_child = buildRecursive(right_indices, max_faces_per_leaf);
  
  nodes_[node_idx].sphere = sphere;
  nodes_[node_idx].left_child = left_child;
  nodes_[node_idx].right_child = right_child;
  
  return node_idx;
}

Sphere BoundingSphereTree::computeBoundingSphere(const std::vector<uint32_t>& face_indices) const
{
  std::unordered_set<const Eigen::Vector3f*> unique_pts;
  
  for (uint32_t idx : face_indices)
  {
    const Face& face = faces_[idx];
    unique_pts.insert(&face.vertices[0]);
    unique_pts.insert(&face.vertices[1]);
    unique_pts.insert(&face.vertices[2]);
  }
  
  std::vector<Eigen::Vector3f> points;
  points.reserve(unique_pts.size());
  for (const auto* pt : unique_pts)
  {
    points.push_back(*pt);
  }
  
  return welzlSphere(std::move(points));
}

Sphere BoundingSphereTree::welzlSphere(std::vector<Eigen::Vector3f>&& points) const
{
  std::shuffle(points.begin(), points.end(), rng_);
  std::vector<Eigen::Vector3f> R;
  return welzlRecursive(points, R, points.size());
}

Sphere BoundingSphereTree::welzlRecursive(std::vector<Eigen::Vector3f>& P,
                                         std::vector<Eigen::Vector3f>& R,
                                         size_t n) const
{
  if (n == 0 || R.size() == 4) 
  {
    return trivialSphere(R);
  }
  
  size_t idx = n - 1;
  Eigen::Vector3f p = P[idx];
  
  Sphere D = welzlRecursive(P, R, n - 1);
  
  if ((p - D.center).squaredNorm() <= D.radius * D.radius * (1.0f + GEOM_EPS)) 
  {
    return D;
  }
  
  R.push_back(p);
  Sphere result = welzlRecursive(P, R, n - 1);
  R.pop_back();
  
  return result;
}

Sphere BoundingSphereTree::trivialSphere(const std::vector<Eigen::Vector3f>& R) const
{
  if (R.empty()) 
  {
    return Sphere{Eigen::Vector3f::Zero(), 0.0f};
  }
  
  if (R.size() == 1) 
  {
    return Sphere{R[0], 0.0f};
  }
  
  if (R.size() == 2) 
  {
    Eigen::Vector3f center = (R[0] + R[1]) * 0.5f;
    float radius = (R[1] - R[0]).norm() * 0.5f;
    return Sphere{center, radius};
  }
  
  if (R.size() == 3) 
  {
    return sphereFrom3Points(R[0], R[1], R[2]);
  }
  
  return sphereFrom4Points(R[0], R[1], R[2], R[3]);
}

Sphere BoundingSphereTree::sphereFrom3Points(const Eigen::Vector3f& a,
                                            const Eigen::Vector3f& b,
                                            const Eigen::Vector3f& c) const
{
  Eigen::Vector3f ab = b - a;
  Eigen::Vector3f ac = c - a;
  Eigen::Vector3f cross = ab.cross(ac);
  
  if (cross.squaredNorm() < GEOM_EPS * GEOM_EPS) 
  {
    float d_ab = ab.squaredNorm();
    float d_ac = ac.squaredNorm();
    float d_bc = (c - b).squaredNorm();
    
    if (d_ab >= d_ac && d_ab >= d_bc) 
    {
      return Sphere{(a + b) * 0.5f, std::sqrt(d_ab) * 0.5f};
    } 
    else if (d_ac >= d_bc) 
    {
      return Sphere{(a + c) * 0.5f, std::sqrt(d_ac) * 0.5f};
    } 
    else 
    {
      return Sphere{(b + c) * 0.5f, std::sqrt(d_bc) * 0.5f};
    }
  }
  
  float d = 2.0f * cross.squaredNorm();
  
  float ab_len_sq = ab.squaredNorm();
  float ac_len_sq = ac.squaredNorm();
  
  Eigen::Vector3f center = a + (ac_len_sq * cross.cross(ab) + ab_len_sq * (-cross).cross(ac)) / d;
  
  float radius = (center - a).norm();
  return Sphere{center, radius};
}

Sphere BoundingSphereTree::sphereFrom4Points(const Eigen::Vector3f& a,
                                            const Eigen::Vector3f& b,
                                            const Eigen::Vector3f& c,
                                            const Eigen::Vector3f& d) const
{
  Eigen::Matrix4f A;
  A << a.x(), a.y(), a.z(), 1.0f,
       b.x(), b.y(), b.z(), 1.0f,
       c.x(), c.y(), c.z(), 1.0f,
       d.x(), d.y(), d.z(), 1.0f;
  
  float det_a = A.determinant();
  
  if (std::abs(det_a) < GEOM_EPS) 
  {
    return sphereFrom3Points(a, b, c);
  }
  
  Eigen::Vector4f rhs;
  rhs << a.squaredNorm(), b.squaredNorm(), c.squaredNorm(), d.squaredNorm();
  
  Eigen::Matrix4f Dx = A;
  Dx.col(0) = rhs;
  float cx = 0.5f * Dx.determinant() / det_a;
  
  Eigen::Matrix4f Dy = A;
  Dy.col(1) = rhs;
  float cy = 0.5f * Dy.determinant() / det_a;
  
  Eigen::Matrix4f Dz = A;
  Dz.col(2) = rhs;
  float cz = 0.5f * Dz.determinant() / det_a;
  
  Eigen::Vector3f center(cx, cy, cz);
  float radius = (center - a).norm();
  
  return Sphere{center, radius};
}

bool BoundingSphereTree::isPointInside(const Eigen::Vector3f& point) const
{
  if (root_node_ == INVALID_ID)
  {
    return false;
  }

  const std::vector<Eigen::Vector3f> directions = {
    Eigen::Vector3f(1.0f, 0.3f, 0.7f).normalized(),
    Eigen::Vector3f(-0.5f, 1.0f, -0.2f).normalized(),
    Eigen::Vector3f(0.1f, -0.8f, 1.0f).normalized()
  };

  for (const auto& direction : directions)
  {
    int intersection_count = 0;
    countRayIntersections(root_node_, point, direction, intersection_count);
    if ((intersection_count % 2) == 1)
    {
      return true;
    }
  }

  return false;
}

void BoundingSphereTree::countRayIntersections(uint32_t node_idx,
                                               const Eigen::Vector3f& origin,
                                               const Eigen::Vector3f& direction,
                                               int& count) const
{
  const BSTNode& node = nodes_[node_idx];
  
  LineSegment ray;
  ray.start = origin;
  ray.end = origin + direction * 1000.0f;
  
  if (!intersectSegmentSphere(ray, node.sphere))
  {
    return;
  }
  
  if (node.isLeaf())
  {
    for (uint32_t face_idx : node.face_indices)
    {
      const Face& face = faces_[face_idx];
      
      float t;
      if (rayTriangleIntersect(origin, direction, face.vertices[0], 
                               face.vertices[1], face.vertices[2], t) && 
          t > GEOM_EPS)
      {
        count++;
        spdlog::debug("Ray hit face {} at t={}", face_idx, t);
      }
    }
    return;
  }
  
  if (node.left_child != INVALID_ID)
  {
    countRayIntersections(node.left_child, origin, direction, count);
  }
  if (node.right_child != INVALID_ID)
  {
    countRayIntersections(node.right_child, origin, direction, count);
  }
}

bool BoundingSphereTree::intersectSphere(const Sphere& sphere) const
{
  if (root_node_ == INVALID_ID) 
  {
    return false;
  }
  
  spdlog::info("intersectSphere called: center=({},{},{}), radius={}", 
               sphere.center.x(), sphere.center.y(), sphere.center.z(), sphere.radius);
  
  bool surface_intersection = intersectSphereRecursive(root_node_, sphere);
  spdlog::info("surface_intersection = {}", surface_intersection);
  
  if (surface_intersection)
  {
    return true;
  }
  
  bool inside = isPointInside(sphere.center);
  spdlog::info("isPointInside = {}", inside);
  
  return inside;
}

bool BoundingSphereTree::intersectSphereRecursive(uint32_t node_idx, const Sphere& sphere) const
{
  if (node_idx >= nodes_.size()) 
  {
    return false;
  }
  
  const BSTNode& node = nodes_[node_idx];
  
  if (!intersectSphereSphere(node.sphere, sphere)) 
  {
    return false;
  }
  
  if (node.isLeaf()) 
  {
    for (uint32_t face_idx : node.face_indices) 
    {
      if (face_idx < faces_.size() && intersectTriangleSphere(faces_[face_idx], sphere)) 
      {
        return true;
      }
    }
    return false;
  }
  
  if (node.left_child != INVALID_ID && node.left_child < nodes_.size() &&
      intersectSphereRecursive(node.left_child, sphere)) 
  {
    return true;
  }
  if (node.right_child != INVALID_ID && node.right_child < nodes_.size() &&
      intersectSphereRecursive(node.right_child, sphere)) 
  {
    return true;
  }
  
  return false;
}

bool BoundingSphereTree::intersectCapsule(const Capsule& capsule) const
{
  if (root_node_ == INVALID_ID) return false;
  
  spdlog::debug("BST::intersectCapsule: p0=({},{},{}), p1=({},{},{}), r={}",
                capsule.p0.x(), capsule.p0.y(), capsule.p0.z(),
                capsule.p1.x(), capsule.p1.y(), capsule.p1.z(), capsule.radius);
  
  bool surface_hit = intersectCapsuleRecursive(root_node_, capsule);
  bool inside = isPointInside(capsule.p0) || isPointInside(capsule.p1);
  
  spdlog::debug("BST: surface_hit={}, inside={}", surface_hit, inside);
  
  return surface_hit || inside;
}

bool BoundingSphereTree::intersectCapsule(const Capsule& capsule, 
                                         collision::HitResult& hit_result) const
{
  if (root_node_ == INVALID_ID) 
  {
    return false;
  }
  
  hit_result.hit = false;
  hit_result.t_min = std::numeric_limits<float>::max();
  
  intersectCapsuleWithHitRecursive(root_node_, capsule, hit_result);
  
  if (hit_result.hit) 
  {
    return true;
  }
  
  if (isPointInside(capsule.p0)) 
  {
    hit_result.hit = true;
    hit_result.object_type = collision::HitResult::ObjectType::STATIC_OBSTACLE;
    
    if (!faces_.empty())
    {
        hit_result.object_id = faces_[0].object_id;
    }
    else
    {
        hit_result.object_id = INVALID_ID;
    }
    
    hit_result.hit_point = capsule.p0;
    hit_result.hit_normal = (capsule.p0 - nodes_[root_node_].sphere.center).normalized();
    hit_result.t_min = 0.0f;
    
    spdlog::debug("BST: Capsule start point is inside mesh, object_id={}", hit_result.object_id);
    
    return true;
  }
  
  return false;
}

void BoundingSphereTree::intersectCapsuleWithHitRecursive(uint32_t node_idx, 
                                                         const Capsule& capsule,
                                                         collision::HitResult& hit_result) const
{
  const BSTNode& node = nodes_[node_idx];
  
  Capsule node_capsule;
  node_capsule.p0 = node_capsule.p1 = node.sphere.center;
  node_capsule.radius = node.sphere.radius;
  
  if (!intersectCapsuleCapsule(capsule, node_capsule)) 
  {
    return;
  }
  
  if (node.isLeaf()) 
  {
    for (uint32_t face_idx : node.face_indices) 
    {
      const Face& face = faces_[face_idx];
      
      Eigen::Vector3f closest_on_tri;
      float t;
      if (intersectTriangleCapsuleWithContact(face, capsule, closest_on_tri, t)) 
      {
        if (t < hit_result.t_min) 
        {
          hit_result.hit = true;
          hit_result.t_min = t;
          hit_result.hit_point = closest_on_tri;
          hit_result.hit_normal = face.normal;
          hit_result.object_id = face.object_id;
          hit_result.object_type = collision::HitResult::ObjectType::STATIC_OBSTACLE;
        }
      }
    }
    return;
  }
  
  if (node.left_child != INVALID_ID) 
  {
    intersectCapsuleWithHitRecursive(node.left_child, capsule, hit_result);
  }
  if (node.right_child != INVALID_ID) 
  {
    intersectCapsuleWithHitRecursive(node.right_child, capsule, hit_result);
  }
}

bool BoundingSphereTree::intersectCapsuleRecursive(uint32_t node_idx, const Capsule& capsule) const
{
  const BSTNode& node = nodes_[node_idx];
  
  Capsule node_capsule;
  node_capsule.p0 = node_capsule.p1 = node.sphere.center;
  node_capsule.radius = node.sphere.radius;
  
  if (!intersectCapsuleCapsule(capsule, node_capsule)) 
  {
    return false;
  }
  
  if (node.isLeaf()) 
  {
    for (uint32_t face_idx : node.face_indices) 
    {
      if (intersectTriangleCapsule(faces_[face_idx], capsule)) 
      {
        return true;
      }
    }
    return false;
  }
  
  if (node.left_child != INVALID_ID && intersectCapsuleRecursive(node.left_child, capsule)) 
  {
    return true;
  }
  if (node.right_child != INVALID_ID && intersectCapsuleRecursive(node.right_child, capsule)) 
  {
    return true;
  }
  
  return false;
}

bool BoundingSphereTree::intersectSegment(const LineSegment& segment) const
{
  Capsule seg_capsule;
  seg_capsule.p0 = segment.start;
  seg_capsule.p1 = segment.end;
  seg_capsule.radius = 0.0f;
  return intersectCapsule(seg_capsule);
}

float BoundingSphereTree::getDistance(const Eigen::Vector3f& point) const
{
  if (root_node_ == INVALID_ID) 
  {
    return std::numeric_limits<float>::max();
  }

  if (isPointInside(point))
  {
    return 0.0f;
  }

  return getDistanceRecursive(root_node_, point, std::numeric_limits<float>::max());
}

float BoundingSphereTree::getDistanceRecursive(uint32_t node_idx, 
                                              const Eigen::Vector3f& point,
                                              float best_so_far) const
{
  const BSTNode& node = nodes_[node_idx];
  
  float node_bound = (point - node.sphere.center).norm() - node.sphere.radius;
  
  if (node_bound >= best_so_far) 
  {
    return std::numeric_limits<float>::max();
  }
  
  if (node.isLeaf()) 
  {
    float min_dist = best_so_far;
    for (uint32_t face_idx : node.face_indices) 
    {
      float dist = distanceToTriangle(faces_[face_idx], point);
      min_dist = std::min(min_dist, dist);
    }
    return min_dist;
  }
  
  float left_bound = std::numeric_limits<float>::max();
  float right_bound = std::numeric_limits<float>::max();
  
  if (node.left_child != INVALID_ID) 
  {
    const Sphere& left_sphere = nodes_[node.left_child].sphere;
    left_bound = (point - left_sphere.center).norm() - left_sphere.radius;
  }
  
  if (node.right_child != INVALID_ID) 
  {
    const Sphere& right_sphere = nodes_[node.right_child].sphere;
    right_bound = (point - right_sphere.center).norm() - right_sphere.radius;
  }
  
  float best = best_so_far;
  
  if (left_bound < right_bound) 
  {
    if (node.left_child != INVALID_ID && left_bound < best) 
    {
      best = std::min(best, getDistanceRecursive(node.left_child, point, best));
    }
    if (node.right_child != INVALID_ID && right_bound < best) 
    {
      best = std::min(best, getDistanceRecursive(node.right_child, point, best));
    }
  } 
  else 
  {
    if (node.right_child != INVALID_ID && right_bound < best) 
    {
      best = std::min(best, getDistanceRecursive(node.right_child, point, best));
    }
    if (node.left_child != INVALID_ID && left_bound < best) 
    {
      best = std::min(best, getDistanceRecursive(node.left_child, point, best));
    }
  }
  
  return best;
}

bool BoundingSphereTree::intersectTriangleSphere(const Face& face, const Sphere& sphere) const
{
  Eigen::Vector3f closest = closestPointOnTriangle(face.vertices[0], face.vertices[1], 
                                                  face.vertices[2], sphere.center);
  
  float dist_sq = (closest - sphere.center).squaredNorm();
  return dist_sq <= sphere.radius * sphere.radius;
}

bool BoundingSphereTree::intersectTriangleCapsule(const Face& face, const Capsule& capsule) const
{
  float len_sq = (capsule.p1 - capsule.p0).squaredNorm();
  if (len_sq < GEOM_EPS * GEOM_EPS) 
  {
    Sphere sphere{capsule.p0, capsule.radius};
    return intersectTriangleSphere(face, sphere);
  }
  
  Sphere s0{capsule.p0, capsule.radius};
  Sphere s1{capsule.p1, capsule.radius};
  
  if (intersectTriangleSphere(face, s0) || intersectTriangleSphere(face, s1)) 
  {
    return true;
  }
  
  Eigen::Vector3f dir = capsule.p1 - capsule.p0;
  float len = dir.norm();
  dir /= len;
  
  float t;
  if (rayTriangleIntersect(capsule.p0, dir, face.vertices[0], face.vertices[1], 
                          face.vertices[2], t) && t >= 0.0f && t <= len) 
  {
    return true;
  }
  
  LineSegment edges[3] = 
  {
    {face.vertices[0], face.vertices[1]},
    {face.vertices[1], face.vertices[2]},
    {face.vertices[2], face.vertices[0]}
  };
  
  for (const auto& edge : edges) 
  {
    float s, t;
    closestPointsOnSegments(edge.start, edge.end, capsule.p0, capsule.p1, s, t);
    
    Eigen::Vector3f p1 = edge.start + s * (edge.end - edge.start);
    Eigen::Vector3f p2 = capsule.p0 + t * (capsule.p1 - capsule.p0);
    
    if ((p1 - p2).squaredNorm() <= capsule.radius * capsule.radius) 
    {
      return true;
    }
  }
  
  return false;
}

bool BoundingSphereTree::intersectTriangleCapsuleWithContact(const Face& face, const Capsule& capsule,
                                                            Eigen::Vector3f& contact_point, float& t) const
{
  float min_dist_sq = std::numeric_limits<float>::max();
  Eigen::Vector3f best_point = face.vertices[0];
  float best_t = 0.0f;
  
  for (int i = 0; i < 3; ++i) 
  {
    Eigen::Vector3f closest;
    float param;
    closestPointOnSegment(capsule.p0, capsule.p1, face.vertices[i], closest, param);
    float dist_sq = (closest - face.vertices[i]).squaredNorm();
    
    if (dist_sq < min_dist_sq) 
    {
      min_dist_sq = dist_sq;
      best_point = face.vertices[i];
      best_t = param;
    }
  }
  
  LineSegment edges[3] = 
  {
    {face.vertices[0], face.vertices[1]},
    {face.vertices[1], face.vertices[2]},
    {face.vertices[2], face.vertices[0]}
  };
  
  for (const auto& edge : edges) 
  {
    float s, t_param;
    closestPointsOnSegments(edge.start, edge.end, capsule.p0, capsule.p1, s, t_param);
    
    Eigen::Vector3f p1 = edge.start + s * (edge.end - edge.start);
    Eigen::Vector3f p2 = capsule.p0 + t_param * (capsule.p1 - capsule.p0);
    float dist_sq = (p1 - p2).squaredNorm();
    
    if (dist_sq < min_dist_sq) 
    {
      min_dist_sq = dist_sq;
      best_point = p1;
      best_t = t_param;
    }
  }
  
  Eigen::Vector3f closest_on_face = closestPointOnTriangle(face.vertices[0], face.vertices[1],
                                                          face.vertices[2], capsule.p0);
  Eigen::Vector3f closest_on_capsule;
  float param;
  closestPointOnSegment(capsule.p0, capsule.p1, closest_on_face, closest_on_capsule, param);
  
  float dist_sq = (closest_on_face - closest_on_capsule).squaredNorm();
  if (dist_sq < min_dist_sq) 
  {
    min_dist_sq = dist_sq;
    best_point = closest_on_face;
    best_t = param;
  }
  
  if (min_dist_sq <= capsule.radius * capsule.radius) 
  {
    contact_point = best_point;
    t = best_t;
    return true;
  }
  
  return false;
}

float BoundingSphereTree::distanceToTriangle(const Face& face, const Eigen::Vector3f& point) const
{
  Eigen::Vector3f closest = closestPointOnTriangle(face.vertices[0], face.vertices[1],
                                                  face.vertices[2], point);
  return (closest - point).norm();
}

Eigen::Vector3f BoundingSphereTree::closestPointOnTriangle(const Eigen::Vector3f& a,
                                                          const Eigen::Vector3f& b,
                                                          const Eigen::Vector3f& c,
                                                          const Eigen::Vector3f& p) const
{
  Eigen::Vector3f ab = b - a;
  Eigen::Vector3f ac = c - a;
  Eigen::Vector3f ap = p - a;
  
  float d1 = ab.dot(ap);
  float d2 = ac.dot(ap);
  if (d1 <= 0.0f && d2 <= 0.0f) 
  {
    return a;
  }
  
  Eigen::Vector3f bp = p - b;
  float d3 = ab.dot(bp);
  float d4 = ac.dot(bp);
  if (d3 >= 0.0f && d4 <= d3) 
  {
    return b;
  }
  
  float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) 
  {
    float v = d1 / (d1 - d3);
    return a + v * ab;
  }
  
  Eigen::Vector3f cp = p - c;
  float d5 = ab.dot(cp);
  float d6 = ac.dot(cp);
  if (d6 >= 0.0f && d5 <= d6) 
  {
    return c;
  }
  
  float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) 
  {
    float w = d2 / (d2 - d6);
    return a + w * ac;
  }
  
  float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) 
  {
    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    return b + w * (c - b);
  }
  
  float denom = 1.0f / (va + vb + vc);
  float v = vb * denom;
  float w = vc * denom;
  return a + ab * v + ac * w;
}

void BoundingSphereTree::closestPointOnSegment(const Eigen::Vector3f& p0,
                                              const Eigen::Vector3f& p1,
                                              const Eigen::Vector3f& point,
                                              Eigen::Vector3f& closest,
                                              float& t) const
{
  Eigen::Vector3f d = p1 - p0;
  float len_sq = d.squaredNorm();
  
  if (len_sq < GEOM_EPS * GEOM_EPS) 
  {
    closest = p0;
    t = 0.0f;
    return;
  }
  
  t = (point - p0).dot(d) / len_sq;
  t = std::max(0.0f, std::min(1.0f, t));
  closest = p0 + t * d;
}

void BoundingSphereTree::closestPointsOnSegments(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1,
                                                const Eigen::Vector3f& q0, const Eigen::Vector3f& q1,
                                                float& s, float& t) const
{
  Eigen::Vector3f u = p1 - p0;
  Eigen::Vector3f v = q1 - q0;
  Eigen::Vector3f w = p0 - q0;
  
  float a = u.dot(u);
  float b = u.dot(v);
  float c = v.dot(v);
  float d = u.dot(w);
  float e = v.dot(w);
  
  float denom = a * c - b * b;
  
  if (std::abs(denom) < GEOM_EPS) 
  {
    s = 0.0f;
    t = (b > c ? d / b : e / c);
  } 
  else 
  {
    s = (b * e - c * d) / denom;
    t = (a * e - b * d) / denom;
  }
  
  s = std::max(0.0f, std::min(1.0f, s));
  t = std::max(0.0f, std::min(1.0f, t));
  
  if (s == 0.0f || s == 1.0f) 
  {
    t = std::max(0.0f, std::min(1.0f, (s * b - e) / c));
  } 
  else if (t == 0.0f || t == 1.0f) 
  {
    s = std::max(0.0f, std::min(1.0f, (t * b + d) / a));
  }
}

bool BoundingSphereTree::rayTriangleIntersect(const Eigen::Vector3f& origin,
                                             const Eigen::Vector3f& dir,
                                             const Eigen::Vector3f& v0,
                                             const Eigen::Vector3f& v1,
                                             const Eigen::Vector3f& v2,
                                             float& t) const
{
  Eigen::Vector3f e1 = v1 - v0;
  Eigen::Vector3f e2 = v2 - v0;
  Eigen::Vector3f h = dir.cross(e2);
  float det = e1.dot(h);
  
  if (std::abs(det) < GEOM_EPS)
  {
    return false;
  }
  
  float inv_det = 1.0f / det;
  Eigen::Vector3f s = origin - v0;
  float u = inv_det * s.dot(h);
  
  if (u < 0.0f || u > 1.0f)
  {
    return false;
  }
  
  Eigen::Vector3f q = s.cross(e1);
  float v = inv_det * dir.dot(q);
  
  if (v < 0.0f || u + v > 1.0f)
  {
    return false;
  }
  
  t = inv_det * e2.dot(q);
  return t > GEOM_EPS;
}

BoundingSphereTree::Stats BoundingSphereTree::getStats() const
{
  Stats stats{};
  stats.total_nodes = nodes_.size();
  stats.leaf_nodes = 0;
  stats.max_depth = 0;
  stats.average_faces_per_leaf = 0.0f;
  
  if (root_node_ == INVALID_ID) 
  {
    return stats;
  }
  
  struct StackEntry 
  {
    uint32_t node_idx;
    size_t depth;
  };
  
  std::vector<StackEntry> stack;
  stack.push_back({root_node_, 0});
  
  size_t total_faces_in_leaves = 0;
  
  while (!stack.empty()) 
  {
    StackEntry entry = stack.back();
    stack.pop_back();
    
    const BSTNode& node = nodes_[entry.node_idx];
    
    if (node.isLeaf()) 
    {
      stats.leaf_nodes++;
      total_faces_in_leaves += node.face_indices.size();
      stats.max_depth = std::max(stats.max_depth, entry.depth);
    } 
    else 
    {
      if (node.left_child != INVALID_ID) 
      {
        stack.push_back({node.left_child, entry.depth + 1});
      }
      if (node.right_child != INVALID_ID) 
      {
        stack.push_back({node.right_child, entry.depth + 1});
      }
    }
  }
  
  if (stats.leaf_nodes > 0) 
  {
    stats.average_faces_per_leaf = static_cast<float>(total_faces_in_leaves) / stats.leaf_nodes;
  }
  
  return stats;
}

void BoundingSphereTree::exportToJSON(const std::string& filename) const
{
  nlohmann::json j;
  
  j["stats"] = 
  {
    {"total_nodes", nodes_.size()},
    {"total_faces", faces_.size()},
    {"root_node", root_node_}
  };
  
  Stats stats = getStats();
  j["stats"]["leaf_nodes"] = stats.leaf_nodes;
  j["stats"]["max_depth"] = stats.max_depth;
  j["stats"]["average_faces_per_leaf"] = stats.average_faces_per_leaf;
  
  nlohmann::json nodes_json = nlohmann::json::array();
  for (size_t i = 0; i < nodes_.size(); ++i) 
  {
    const BSTNode& node = nodes_[i];
    nlohmann::json node_json;
    
    node_json["index"] = i;
    node_json["sphere"] = 
    {
      {"center", {node.sphere.center.x(), node.sphere.center.y(), node.sphere.center.z()}},
      {"radius", node.sphere.radius}
    };
    node_json["is_leaf"] = node.isLeaf();
    
    if (node.isLeaf()) 
    {
      node_json["face_count"] = node.face_indices.size();
      node_json["face_indices"] = node.face_indices;
    } 
    else 
    {
      node_json["left_child"] = node.left_child;
      node_json["right_child"] = node.right_child;
    }
    
    nodes_json.push_back(node_json);
  }
  j["nodes"] = nodes_json;
  
  nlohmann::json faces_json = nlohmann::json::array();
  size_t max_faces_to_export = 1000;
  size_t step = faces_.size() > max_faces_to_export ? faces_.size() / max_faces_to_export : 1;
  
  for (size_t i = 0; i < faces_.size(); i += step) 
  {
    const Face& face = faces_[i];
    nlohmann::json face_json;
    
    face_json["index"] = i;
    face_json["vertices"] = nlohmann::json::array();
    for (int v = 0; v < 3; ++v) 
    {
      face_json["vertices"].push_back({face.vertices[v].x(), face.vertices[v].y(), face.vertices[v].z()});
    }
    face_json["normal"] = {face.normal.x(), face.normal.y(), face.normal.z()};
    face_json["object_id"] = face.object_id;
    
    faces_json.push_back(face_json);
  }
  j["faces_sample"] = faces_json;
  j["total_faces"] = faces_.size();
  
  std::ofstream file(filename);
  file << j.dump(2);
}

}
}
}