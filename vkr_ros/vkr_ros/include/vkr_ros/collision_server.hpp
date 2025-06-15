#ifndef VKR_ROS_COLLISION_SERVER_HPP_
#define VKR_ROS_COLLISION_SERVER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vkr_srvs/srv/check_capsule.hpp>
#include <vkr_srvs/srv/check_segment.hpp>
#include <vkr_srvs/srv/check_path.hpp>
#include <vkr_srvs/srv/get_distance.hpp>
#include <vkr/geometry/collision/collision_query.hpp>
#include <vkr/terrain/wavelet_grid.hpp>
#include <vkr/geometry/static_obstacles/bounding_sphere_tree.hpp>
#include <vkr/geometry/corridors/spatial_corridor_map.hpp>
#include <vkr/config.hpp>
#include <memory>
#include <chrono>

namespace vkr_ros
{

class CollisionServer : public rclcpp::Node
{
public:
  explicit CollisionServer(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~CollisionServer() = default;

private:
  void loadTerrain(const std::string& terrain_file);
  void convertGeoTIFFToQWG(const std::string& geotiff_file, const std::string& qwg_file);
  void loadStaticObstacles(const std::vector<std::string>& obstacle_files,
                          const std::vector<std::vector<float>>& obstacle_transforms);
  
  void handleCheckCapsule(
      const std::shared_ptr<vkr_srvs::srv::CheckCapsule::Request> request,
      std::shared_ptr<vkr_srvs::srv::CheckCapsule::Response> response);
  
  void handleCheckSegment(
      const std::shared_ptr<vkr_srvs::srv::CheckSegment::Request> request,
      std::shared_ptr<vkr_srvs::srv::CheckSegment::Response> response);
  
  void handleCheckPath(
      const std::shared_ptr<vkr_srvs::srv::CheckPath::Request> request,
      std::shared_ptr<vkr_srvs::srv::CheckPath::Response> response);
  
  void handleGetDistance(
      const std::shared_ptr<vkr_srvs::srv::GetDistance::Request> request,
      std::shared_ptr<vkr_srvs::srv::GetDistance::Response> response);
  
  std::chrono::steady_clock::time_point rosTimeToSteady(const builtin_interfaces::msg::Time& ros_time) const;
  std::string objectTypeToString(vkr::geometry::collision::HitResult::ObjectType type) const;
  
  std::unique_ptr<vkr::geometry::collision::CollisionQuery> collision_query_;
  std::shared_ptr<vkr::terrain::WaveletGrid> terrain_;
  std::shared_ptr<vkr::geometry::static_obstacles::BoundingSphereTree> static_obstacles_;
  
  rclcpp::Service<vkr_srvs::srv::CheckCapsule>::SharedPtr check_capsule_srv_;
  rclcpp::Service<vkr_srvs::srv::CheckSegment>::SharedPtr check_segment_srv_;
  rclcpp::Service<vkr_srvs::srv::CheckPath>::SharedPtr check_path_srv_;
  rclcpp::Service<vkr_srvs::srv::GetDistance>::SharedPtr get_distance_srv_;
  
  vkr::VkrConfig config_;
  mutable std::mutex collision_mutex_;
};

}

#endif