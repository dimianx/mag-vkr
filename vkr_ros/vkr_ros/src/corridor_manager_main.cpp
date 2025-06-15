#include "vkr_ros/corridor_manager.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  
  auto node = std::make_shared<vkr_ros::CorridorManager>(options);
  
  RCLCPP_INFO(node->get_logger(), "Starting Corridor Manager...");
  
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  
  try {
    executor.spin();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Exception in corridor manager: %s", e.what());
    return 1;
  }
  
  rclcpp::shutdown();
  return 0;
}