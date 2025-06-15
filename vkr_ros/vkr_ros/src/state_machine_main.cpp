#include "vkr_ros/state_machine.hpp"
#include <rclcpp/rclcpp.hpp>
#include <memory>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  rclcpp::NodeOptions options;
  options.use_intra_process_comms(true);
  
  if (argc > 1) {
    options.arguments({"--ros-args", "-r", "__ns:=" + std::string(argv[1])});
  }
  
  auto node = std::make_shared<vkr_ros::StateMachine>(options);
  
  RCLCPP_INFO(node->get_logger(), "Starting State Machine for namespace: %s", 
              node->get_namespace());
  
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  
  try {
    executor.spin();
  } catch (const std::exception& e) {
    RCLCPP_ERROR(node->get_logger(), "Exception in state machine: %s", e.what());
    return 1;
  }
  
  rclcpp::shutdown();
  return 0;
}