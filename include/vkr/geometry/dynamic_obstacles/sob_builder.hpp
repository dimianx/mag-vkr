#ifndef VKR_GEOMETRY_DYNAMIC_OBSTACLES_SOB_BUILDER_HPP_
#define VKR_GEOMETRY_DYNAMIC_OBSTACLES_SOB_BUILDER_HPP_

#include "vkr/geometry/dynamic_obstacles/swept_obb_bvh.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <string>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

OBB computeOptimalOBB(const std::vector<Eigen::Vector3f>& points);

class SweptOBBBVHBuilder
{
public:
  explicit SweptOBBBVHBuilder(const GeometryConfig& config);
  ~SweptOBBBVHBuilder();
  
  SweptOBBBVHBuilder& addFromMesh(
      const std::string& filename,
      ObjectId object_id,
      const std::vector<Eigen::Vector3f>& position_trajectory,
      const std::vector<Eigen::Vector3f>& rotation_trajectory,
      const std::chrono::steady_clock::time_point& t_start,
      const std::chrono::steady_clock::time_point& t_end);
  
  SweptOBBBVHBuilder& addWithTimeMapping(
      const OBB& obb,
      ObjectId object_id,
      const std::vector<Eigen::Vector3f>& position_trajectory,
      const std::vector<Eigen::Vector3f>& rotation_trajectory,
      const std::vector<float>& time_coeffs,
      const std::chrono::steady_clock::time_point& t_start,
      const std::chrono::steady_clock::time_point& t_end);
  
  SweptOBBBVHBuilder& setTrajectoryDegree(int degree);
  
  std::unique_ptr<SweptOBBBVH> build();
  
  SweptOBBBVHBuilder& clear();
  
  size_t getObstacleCount() const;
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  GeometryConfig config_;
};

}
}
}

#endif