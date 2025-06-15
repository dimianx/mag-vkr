#include "vkr_ros/collision_server.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  
  auto node = std::make_shared<vkr_ros::CollisionServer>(options);
  
  RCLCPP_INFO(node->get_logger(), "Starting Collision Server...");
  
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  
  try {
    executor.spin();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Exception in collision server: %s", e.what());
    return 1;
  }
  
  rclcpp::shutdown();
  return 0;
}