#ifndef VKR_ROS_CORRIDOR_MANAGER_HPP_
#define VKR_ROS_CORRIDOR_MANAGER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <vkr_msgs/msg/corridor_free_event.hpp>
#include <vkr_srvs/srv/create_corridor.hpp>
#include <vkr_srvs/srv/update_corridor.hpp>
#include <vkr_srvs/srv/remove_corridor.hpp>
#include <vkr_srvs/srv/check_intrusion.hpp>
#include <vkr/geometry/corridors/spatial_corridor_map.hpp>
#include <vkr/config.hpp>
#include <tbb/concurrent_queue.h>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>

namespace vkr_ros
{

class CorridorManager : public rclcpp::Node
{
public:
  explicit CorridorManager(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~CorridorManager() = default;

private:
  struct UAVState
  {
    vkr::UAVId id;
    Eigen::Vector3f position;
    std::vector<vkr::CorridorId> active_corridors;
    rclcpp::Time last_update;
  };

  void handleCreateCorridor(
      const std::shared_ptr<vkr_srvs::srv::CreateCorridor::Request> request,
      std::shared_ptr<vkr_srvs::srv::CreateCorridor::Response> response);

  void handleUpdateCorridor(
      const std::shared_ptr<vkr_srvs::srv::UpdateCorridor::Request> request,
      std::shared_ptr<vkr_srvs::srv::UpdateCorridor::Response> response);

  void handleRemoveCorridor(
      const std::shared_ptr<vkr_srvs::srv::RemoveCorridor::Request> request,
      std::shared_ptr<vkr_srvs::srv::RemoveCorridor::Response> response);

  void handleCheckIntrusion(
      const std::shared_ptr<vkr_srvs::srv::CheckIntrusion::Request> request,
      std::shared_ptr<vkr_srvs::srv::CheckIntrusion::Response> response);

  void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, vkr::UAVId uav_id);
  
  void checkAndFreeSegments();
  
  void publishQueuedEvents();

  std::shared_ptr<vkr::geometry::corridors::SpatialCorridorMap> spatial_map_;
  
  std::unordered_map<vkr::UAVId, UAVState> uav_states_;
  std::mutex states_mutex_;
  
  mutable std::shared_mutex corridor_operations_mutex_;
  
  tbb::concurrent_queue<vkr_msgs::msg::CorridorFreeEvent> event_queue_;
  
  std::unordered_map<vkr::UAVId, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> position_subs_;
  
  rclcpp::Publisher<vkr_msgs::msg::CorridorFreeEvent>::SharedPtr free_event_pub_;
  
  rclcpp::Service<vkr_srvs::srv::CreateCorridor>::SharedPtr create_corridor_srv_;
  rclcpp::Service<vkr_srvs::srv::UpdateCorridor>::SharedPtr update_corridor_srv_;
  rclcpp::Service<vkr_srvs::srv::RemoveCorridor>::SharedPtr remove_corridor_srv_;
  rclcpp::Service<vkr_srvs::srv::CheckIntrusion>::SharedPtr check_intrusion_srv_;
  
  rclcpp::TimerBase::SharedPtr check_timer_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  
  vkr::VkrConfig config_;
  float corridor_free_distance_;
};

}

#endif