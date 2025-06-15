#ifndef VKR_ROS_STATE_MACHINE_HPP_
#define VKR_ROS_STATE_MACHINE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_bool.hpp>
#include <mavros_msgs/srv/set_mode.hpp>
#include <vkr_msgs/msg/path.hpp>
#include <vkr_msgs/msg/landing_zone.hpp>
#include <vkr_srvs/srv/create_corridor.hpp>
#include <vector>
#include <string>

namespace vkr_ros
{

class StateMachine : public rclcpp::Node
{
public:
  explicit StateMachine(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~StateMachine() = default;

private:
  enum class State
  {
    INIT,
    IDLE,
    TAKEOFF,
    PLANNING,
    FLYING,
    LANDING,
    LANDED,
    RETURN_TO_BASE
  };
  
  struct Mission
  {
    std::vector<Eigen::Vector3f> waypoints;
    size_t current_waypoint;
    Eigen::Vector3f base_position;
  };
  
  void loadMission(const std::string& mission_file);
  void stateCallback(const mavros_msgs::msg::State::SharedPtr msg);
  void positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void pathCallback(const vkr_msgs::msg::Path::SharedPtr msg);
  void landingZoneCallback(const vkr_msgs::msg::LandingZone::SharedPtr msg);
  
  void runStateMachine();
  void transitionTo(State new_state);
  
  void executeInit();
  void executeIdle();
  void executeTakeoff();
  void executePlanning();
  void executeFlying();
  void executeLanding();
  void executeLanded();
  void executeReturnToBase();
  
  bool arm();
  bool disarm();
  bool setOffboardMode();
  bool setLandMode();
  void publishGoal(const Eigen::Vector3f& goal);
  void publishSetpoint(const Eigen::Vector3f& position);
  bool isAtPosition(const Eigen::Vector3f& target, float tolerance = 0.5f);
  void sendOffboardSetpoint();
  
  State current_state_;
  Mission mission_;
  
  Eigen::Vector3f current_position_;
  Eigen::Vector3f current_goal_;
  std::vector<Eigen::Vector3f> current_path_;
  size_t path_index_;
  vkr::CorridorId current_corridor_;
  
  bool is_armed_;
  bool is_offboard_;
  bool has_landing_zone_;
  Eigen::Vector3f landing_zone_center_;

  rclcpp::TimerBase::SharedPtr offboard_timer_;
  Eigen::Vector3f target_position_;
  std::string current_mode_;
  int offboard_stream_rate_hz_;
  
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr position_sub_;
  rclcpp::Subscription<vkr_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<vkr_msgs::msg::LandingZone>::SharedPtr landing_zone_sub_;
  
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr setpoint_pub_;
  
  rclcpp::Client<mavros_msgs::srv::CommandBool>::SharedPtr arming_client_;
  rclcpp::Client<mavros_msgs::srv::SetMode>::SharedPtr set_mode_client_;
  rclcpp::Client<vkr_srvs::srv::CreateCorridor>::SharedPtr create_corridor_client_;
  
  rclcpp::TimerBase::SharedPtr state_timer_;
  
  vkr::UAVId uav_id_;
  float takeoff_altitude_;
  float landing_descent_rate_;
};

}

#endif