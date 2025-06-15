#include "vkr_ros/landing_detector.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace vkr_ros
{

LandingDetector::LandingDetector(const rclcpp::NodeOptions& options)
  : Node("landing_detector", options),
    process_enabled_(false)
{
  declare_parameter("ransac_iterations", 100);
  declare_parameter("ransac_distance_threshold", 0.02);
  declare_parameter("min_points_for_plane", 100);
  declare_parameter("max_tilt_angle", 5.0);
  declare_parameter("min_area", 0.25);
  declare_parameter("max_roughness", 0.02);
  declare_parameter("min_process_altitude", 20.0);
  
  config_.landing.ransac_iterations = get_parameter("ransac_iterations").as_int();
  config_.landing.ransac_distance_threshold = get_parameter("ransac_distance_threshold").as_double();
  config_.landing.min_points_for_plane = get_parameter("min_points_for_plane").as_int();
  config_.landing.max_tilt_angle = get_parameter("max_tilt_angle").as_double();
  config_.landing.min_area = get_parameter("min_area").as_double();
  config_.landing.max_roughness = get_parameter("max_roughness").as_double();
  
  min_process_altitude_ = get_parameter("min_process_altitude").as_double();
  
  landing_unit_ = std::make_unique<vkr::landing::LandingUnit>(config_.landing);
  
  std::string ns = get_namespace();
  
  cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      ns + "/camera/depth/points", 10,
      std::bind(&LandingDetector::pointCloudCallback, this, std::placeholders::_1));
  
  position_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ns + "/mavros/local_position/pose", 10,
      std::bind(&LandingDetector::positionCallback, this, std::placeholders::_1));
  
  landing_zone_pub_ = create_publisher<vkr_msgs::msg::LandingZone>(
      ns + "/landing/best_zone", 10);
  
  RCLCPP_INFO(get_logger(), "Landing detector initialized");
}

void LandingDetector::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (!process_enabled_ || current_position_.z() > min_process_altitude_)
  {
    return;
  }
  
  auto cloud = convertPointCloud(*msg);
  
  if (cloud.points.empty())
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Received empty point cloud");
    return;
  }
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_position_);
  
  publishResults(result);
}

void LandingDetector::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  current_position_ = Eigen::Vector3f(
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z);
  
  process_enabled_ = current_position_.z() <= min_process_altitude_;
}

vkr::landing::PointCloud LandingDetector::convertPointCloud(const sensor_msgs::msg::PointCloud2& msg)
{
  pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
  pcl::fromROSMsg(msg, pcl_cloud);
  
  vkr::landing::PointCloud cloud;
  cloud.points.reserve(pcl_cloud.size());
  
  Eigen::Vector3f min_bound = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f max_bound = Eigen::Vector3f::Constant(-std::numeric_limits<float>::max());
  
  for (const auto& p : pcl_cloud.points)
  {
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
    {
      Eigen::Vector3f point(p.x, p.y, p.z);
      cloud.points.push_back(point);
      
      min_bound = min_bound.cwiseMin(point);
      max_bound = max_bound.cwiseMax(point);
    }
  }
  
  cloud.min_bound = min_bound;
  cloud.max_bound = max_bound;
  
  return cloud;
}

void LandingDetector::publishResults(const vkr::landing::LandingResult& result)
{
  vkr_msgs::msg::LandingZone msg;
  msg.header.stamp = now();
  msg.header.frame_id = "map";
  msg.has_valid_zone = result.has_valid_landing_zone;
  msg.computation_time_ms = result.computation_time_ms;
  
  if (result.has_valid_landing_zone)
  {
    const auto& best = result.best_candidate;
    
    msg.best_candidate.center.x = best.center.x();
    msg.best_candidate.center.y = best.center.y();
    msg.best_candidate.center.z = best.center.z();
    
    msg.best_candidate.normal.x = best.normal.x();
    msg.best_candidate.normal.y = best.normal.y();
    msg.best_candidate.normal.z = best.normal.z();
    
    msg.best_candidate.score = best.score;
    msg.best_candidate.area = best.area;
    msg.best_candidate.roughness = best.roughness;
    msg.best_candidate.tilt_angle = best.tilt_angle;
    msg.best_candidate.confidence = best.confidence;
    
    for (const auto& bp : best.boundary_points)
    {
      geometry_msgs::msg::Point p;
      p.x = bp.x();
      p.y = bp.y();
      p.z = bp.z();
      msg.best_candidate.boundary_points.push_back(p);
    }
  }
  
  for (const auto& candidate : result.candidates)
  {
    vkr_msgs::msg::LandingCandidate c;
    
    c.center.x = candidate.center.x();
    c.center.y = candidate.center.y();
    c.center.z = candidate.center.z();
    
    c.normal.x = candidate.normal.x();
    c.normal.y = candidate.normal.y();
    c.normal.z = candidate.normal.z();
    
    c.score = candidate.score;
    c.area = candidate.area;
    c.roughness = candidate.roughness;
    c.tilt_angle = candidate.tilt_angle;
    c.confidence = candidate.confidence;
    
    msg.all_candidates.push_back(c);
  }
  
  landing_zone_pub_->publish(msg);
  
  if (result.has_valid_landing_zone)
  {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                         "Found landing zone at [%.2f, %.2f, %.2f] with score %.2f",
                         result.best_candidate.center.x(),
                         result.best_candidate.center.y(),
                         result.best_candidate.center.z(),
                         result.best_candidate.score);
  }
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vkr_ros::LandingDetector)