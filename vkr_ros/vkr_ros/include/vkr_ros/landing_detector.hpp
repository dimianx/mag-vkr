#ifndef VKR_ROS_LANDING_DETECTOR_HPP_
#define VKR_ROS_LANDING_DETECTOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vkr_msgs/msg/landing_zone.hpp>
#include <vkr/landing/landing_unit.hpp>
#include <vkr/config.hpp>
#include <memory>

namespace vkr_ros
{

class LandingDetector : public rclcpp::Node
{
public:
  explicit LandingDetector(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~LandingDetector() = default;

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  
  vkr::landing::PointCloud convertPointCloud(const sensor_msgs::msg::PointCloud2& msg);
  void publishResults(const vkr::landing::LandingResult& result);
  
  std::unique_ptr<vkr::landing::LandingUnit> landing_unit_;
  
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
  
  rclcpp::Publisher<vkr_msgs::msg::LandingZone>::SharedPtr landing_zone_pub_;
  
  Eigen::Vector3f current_position_;
  vkr::VkrConfig config_;
  
  bool process_enabled_;
  double min_process_altitude_;
};

}

#endif