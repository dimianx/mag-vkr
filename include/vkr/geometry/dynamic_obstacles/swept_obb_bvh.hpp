#ifndef VKR_GEOMETRY_DYNAMIC_OBSTACLES_SWEPT_OBB_BVH_HPP_
#define VKR_GEOMETRY_DYNAMIC_OBSTACLES_SWEPT_OBB_BVH_HPP_

#include "vkr/geometry/dynamic_obstacles/swept_obb.hpp"
#include "vkr/geometry/collision/hit_result.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <vector>
#include <chrono>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

class SweptOBBBVH
{
public:
  explicit SweptOBBBVH(const GeometryConfig& config);
  ~SweptOBBBVH();
  
  collision::HitResult queryPoint(
      const Eigen::Vector3f& point,
      const std::chrono::steady_clock::time_point& time) const;
  
  collision::HitResult queryOBB(
      const OBB& obb,
      const std::chrono::steady_clock::time_point& t0,
      const std::chrono::steady_clock::time_point& t1) const;
  
  collision::HitResult queryPath(
      const std::vector<Eigen::Vector3f>& waypoints,
      const Eigen::Vector3f& half_extents,
      const std::chrono::steady_clock::time_point& start_time,
      float velocity) const;
  
  void batchQueryPoints(
      const Eigen::Vector3f* points,
      size_t count,
      const std::chrono::steady_clock::time_point& time,
      collision::HitResult* results) const;
  
  std::vector<OBB> getActiveOBBs(
      const std::chrono::steady_clock::time_point& time) const;
  
  float getDistance(
      const Eigen::Vector3f& point,
      const std::chrono::steady_clock::time_point& time) const;
  
  struct Stats
  {
    size_t num_objects = 0;
    size_t num_nodes = 0;
    size_t tree_depth = 0;
    size_t num_leaves = 0;
    float avg_obbs_per_leaf = 0.0f;
    float construction_time_ms = 0.0f;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct BVHNode;
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  friend class SweptOBBBVHBuilder;
  friend void buildBVHFromBuilder(SweptOBBBVH* bvh, std::vector<SweptOBB>&& obbs);
};

}
}
}

#endif