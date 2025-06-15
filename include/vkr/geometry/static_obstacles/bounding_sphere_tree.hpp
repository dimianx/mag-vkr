#ifndef VKR_GEOMETRY_STATIC_OBSTACLES_BOUNDING_SPHERE_TREE_HPP_
#define VKR_GEOMETRY_STATIC_OBSTACLES_BOUNDING_SPHERE_TREE_HPP_

#include "vkr/geometry/static_obstacles/bst_node.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/common.hpp"
#include <memory>
#include <vector>
#include <random>

namespace vkr
{
namespace geometry
{

namespace collision
{
  struct HitResult;
}

namespace static_obstacles
{

class BoundingSphereTree
{
public:
  BoundingSphereTree();
  ~BoundingSphereTree();
  
  struct Face
  {
    Eigen::Vector3f vertices[3];
    Eigen::Vector3f normal;
    ObjectId object_id;
  };
  
  void build(const std::vector<Face>& faces, uint32_t max_faces_per_leaf = 32);
  
  bool intersectSphere(const Sphere& sphere) const;
  bool intersectCapsule(const Capsule& capsule) const;
  bool intersectCapsule(const Capsule& capsule,
                       collision::HitResult& hit_result) const;
  bool intersectSegment(const LineSegment& segment) const;

  bool isPointInside(const Eigen::Vector3f& point) const;
  
  float getDistance(const Eigen::Vector3f& point) const;
    
  struct Stats
  {
    size_t total_nodes;
    size_t leaf_nodes;
    size_t max_depth;
    float average_faces_per_leaf;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  std::vector<BSTNode> nodes_;
  std::vector<Face> faces_;
  uint32_t root_node_ = INVALID_ID;
  mutable std::mt19937 rng_{std::random_device{}()};
  
  uint32_t buildRecursive(std::vector<uint32_t>& face_indices,
                         uint32_t max_faces_per_leaf);
  Sphere computeBoundingSphere(const std::vector<uint32_t>& face_indices) const;
  
  bool intersectSphereRecursive(uint32_t node_idx, const Sphere& sphere) const;
  bool intersectCapsuleRecursive(uint32_t node_idx, const Capsule& capsule) const;
  float getDistanceRecursive(uint32_t node_idx, const Eigen::Vector3f& point, float best_so_far) const;
  
  bool intersectTriangleSphere(const Face& face, const Sphere& sphere) const;
  bool intersectTriangleCapsule(const Face& face, const Capsule& capsule) const;
  float distanceToTriangle(const Face& face, const Eigen::Vector3f& point) const;
  
  Sphere welzlSphere(std::vector<Eigen::Vector3f>&& points) const;
  Sphere welzlRecursive(std::vector<Eigen::Vector3f>& P,
                       std::vector<Eigen::Vector3f>& R,
                       size_t n) const;
  Sphere trivialSphere(const std::vector<Eigen::Vector3f>& R) const;
  Sphere sphereFrom3Points(const Eigen::Vector3f& a,
                          const Eigen::Vector3f& b,
                          const Eigen::Vector3f& c) const;
  Sphere sphereFrom4Points(const Eigen::Vector3f& a,
                          const Eigen::Vector3f& b,
                          const Eigen::Vector3f& c,
                          const Eigen::Vector3f& d) const;
  
  void intersectCapsuleWithHitRecursive(uint32_t node_idx, 
                                       const Capsule& capsule,
                                       collision::HitResult& hit_result) const;
  bool intersectTriangleCapsuleWithContact(const Face& face, const Capsule& capsule,
                                          Eigen::Vector3f& contact_point, float& t) const;
  
  Eigen::Vector3f closestPointOnTriangle(const Eigen::Vector3f& a,
                                        const Eigen::Vector3f& b,
                                        const Eigen::Vector3f& c,
                                        const Eigen::Vector3f& p) const;
  void closestPointOnSegment(const Eigen::Vector3f& p0,
                            const Eigen::Vector3f& p1,
                            const Eigen::Vector3f& point,
                            Eigen::Vector3f& closest,
                            float& t) const;
  void closestPointsOnSegments(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1,
                              const Eigen::Vector3f& q0, const Eigen::Vector3f& q1,
                              float& s, float& t) const;
  bool rayTriangleIntersect(const Eigen::Vector3f& origin,
                           const Eigen::Vector3f& dir,
                           const Eigen::Vector3f& v0,
                           const Eigen::Vector3f& v1,
                           const Eigen::Vector3f& v2,
                           float& t) const;

  void countRayIntersections(uint32_t node_idx,
                            const Eigen::Vector3f& origin,
                            const Eigen::Vector3f& direction,
                            int& count) const;
};

}
}
}

#endif