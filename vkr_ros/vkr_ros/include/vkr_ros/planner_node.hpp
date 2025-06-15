#ifndef VKR_ROS_PLANNER_NODE_HPP_
#define VKR_ROS_PLANNER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vkr_msgs/msg/path.hpp>
#include <vkr_msgs/msg/corridor_free_event.hpp>
#include <vkr_srvs/srv/check_capsule.hpp>
#include <vkr_srvs/srv/create_corridor.hpp>
#include <vkr/planning/amcp_planner.hpp>
#include <vkr/config.hpp>
#include <memory>
#include <future>

namespace vkr_ros
{

class PlannerNode : public rclcpp::Node
{
public:
  explicit PlannerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~PlannerNode() = default;

private:
  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void corridorFreeCallback(const vkr_msgs::msg::CorridorFreeEvent::SharedPtr msg);
  
  void planPath();
  void checkForReplan();
  void publishPath(const vkr::planning::PlanResult& result);
  
  bool acquirePlanningLock();
  void releasePlanningLock();
  
  std::unique_ptr<vkr::planning::AMCPPlanner> planner_;
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
  rclcpp::Subscription<vkr_msgs::msg::CorridorFreeEvent>::SharedPtr corridor_free_sub_;
  
  rclcpp::Publisher<vkr_msgs::msg::Path>::SharedPtr path_pub_;
  
  rclcpp::Client<vkr_srvs::srv::CheckCapsule>::SharedPtr check_capsule_client_;
  rclcpp::Client<vkr_srvs::srv::CreateCorridor>::SharedPtr create_corridor_client_;
  
  Eigen::Vector3f current_position_;
  Eigen::Vector3f goal_position_;
  bool has_goal_;
  bool has_path_;
  
  vkr::planning::PlanResult current_plan_;
  
  vkr::UAVId uav_id_;
  vkr::VkrConfig config_;
  
  std::future<vkr::planning::PlanResult> planning_future_;
  bool planning_in_progress_;
  bool needs_replan_;
  
  rclcpp::TimerBase::SharedPtr replan_timer_;
  
  int planning_timeout_ms_;
};

}

#endif