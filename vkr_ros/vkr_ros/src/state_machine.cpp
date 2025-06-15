#include "vkr_ros/state_machine.hpp"
#include <yaml-cpp/yaml.h>
#include <fstream>

namespace vkr_ros
{

StateMachine::StateMachine(const rclcpp::NodeOptions& options)
  : Node("state_machine", options),
    current_state_(State::INIT),
    path_index_(0),
    current_corridor_(vkr::INVALID_ID),
    is_armed_(false),
    is_offboard_(false),
    has_landing_zone_(false)
{
  declare_parameter("uav_id", 1);
  declare_parameter("mission_file", "");
  declare_parameter("takeoff_altitude", 10.0);
  declare_parameter("landing_descent_rate", 0.5);
  declare_parameter("offboard_stream_rate_hz", 20);
  
  uav_id_ = get_parameter("uav_id").as_int();
  takeoff_altitude_ = get_parameter("takeoff_altitude").as_double();
  landing_descent_rate_ = get_parameter("landing_descent_rate").as_double();
  offboard_stream_rate_hz_ = get_parameter("offboard_stream_rate_hz").as_int();
  
  std::string mission_file = get_parameter("mission_file").as_string();
  if (!mission_file.empty())
  {
    loadMission(mission_file);
  }
  
  std::string ns = get_namespace();
  
  state_sub_ = create_subscription<mavros_msgs::msg::State>(
      ns + "/mavros/state", 10,
      std::bind(&StateMachine::stateCallback, this, std::placeholders::_1));
  
  position_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ns + "/mavros/local_position/pose", 10,
      std::bind(&StateMachine::positionCallback, this, std::placeholders::_1));
  
  path_sub_ = create_subscription<vkr_msgs::msg::Path>(
      ns + "/planned_path", 10,
      std::bind(&StateMachine::pathCallback, this, std::placeholders::_1));
  
  landing_zone_sub_ = create_subscription<vkr_msgs::msg::LandingZone>(
      ns + "/landing/best_zone", 10,
      std::bind(&StateMachine::landingZoneCallback, this, std::placeholders::_1));
  
  goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(ns + "/goal_pose", 10);
  setpoint_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
      ns + "/mavros/setpoint_position/local", 10);
  
  arming_client_ = create_client<mavros_msgs::srv::CommandBool>(ns + "/mavros/cmd/arming");
  set_mode_client_ = create_client<mavros_msgs::srv::SetMode>(ns + "/mavros/set_mode");
  create_corridor_client_ = create_client<vkr_srvs::srv::CreateCorridor>("/corridors/create");
  
  state_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&StateMachine::runStateMachine, this));
  
  offboard_timer_ = create_wall_timer(
      std::chrono::milliseconds(1000 / offboard_stream_rate_hz_),
      std::bind(&StateMachine::sendOffboardSetpoint, this));
  
  RCLCPP_INFO(get_logger(), "State machine initialized for UAV %d", uav_id_);
}

void StateMachine::loadMission(const std::string& mission_file)
{
  try
  {
    YAML::Node config = YAML::LoadFile(mission_file);
    
    mission_.base_position = Eigen::Vector3f(
        config["base"]["x"].as<float>(),
        config["base"]["y"].as<float>(),
        config["base"]["z"].as<float>());
    
    mission_.waypoints.clear();
    for (const auto& wp : config["waypoints"])
    {
      mission_.waypoints.emplace_back(
          wp["x"].as<float>(),
          wp["y"].as<float>(),
          wp["z"].as<float>());
    }
    
    mission_.current_waypoint = 0;
    
    RCLCPP_INFO(get_logger(), "Loaded mission with %zu waypoints", mission_.waypoints.size());
  }
  catch (const std::exception& e)
  {
    RCLCPP_ERROR(get_logger(), "Failed to load mission: %s", e.what());
  }
}

void StateMachine::stateCallback(const mavros_msgs::msg::State::SharedPtr msg)
{
  is_armed_ = msg->armed;
  is_offboard_ = msg->mode == "OFFBOARD";
  current_mode_ = msg->mode;
}

void StateMachine::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  current_position_ = Eigen::Vector3f(
      msg->pose.position.x,
      msg->pose.position.y,
      msg->pose.position.z);
}

void StateMachine::pathCallback(const vkr_msgs::msg::Path::SharedPtr msg)
{
  current_path_.clear();
  for (const auto& wp : msg->waypoints)
  {
    current_path_.emplace_back(wp.x, wp.y, wp.z);
  }
  
  if (!current_path_.empty() && current_state_ == State::PLANNING)
  {
    auto request = std::make_shared<vkr_srvs::srv::CreateCorridor::Request>();
    request->uav_id = uav_id_;
    request->path = msg->waypoints;
    request->radius = 2.5f;
    request->current_waypoint_index = 0;
    
    auto future = create_corridor_client_->async_send_request(request);
    
    if (rclcpp::spin_until_future_complete(shared_from_this(), future, std::chrono::seconds(5)) == 
        rclcpp::FutureReturnCode::SUCCESS)
    {
      auto result = future.get();
      if (result->success)
      {
        current_corridor_ = result->corridor_id;
        path_index_ = 0;
        transitionTo(State::FLYING);
        RCLCPP_INFO(get_logger(), "Created corridor %u, starting flight", current_corridor_);
      }
      else
      {
        RCLCPP_ERROR(get_logger(), "Failed to create corridor: %s", result->error_message.c_str());
        current_path_.clear();
      }
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Corridor creation service timeout");
      current_path_.clear();
    }
  }
}

void StateMachine::landingZoneCallback(const vkr_msgs::msg::LandingZone::SharedPtr msg)
{
  if (msg->has_valid_zone)
  {
    has_landing_zone_ = true;
    landing_zone_center_ = Eigen::Vector3f(
        msg->best_candidate.center.x,
        msg->best_candidate.center.y,
        msg->best_candidate.center.z);
  }
}

void StateMachine::runStateMachine()
{
  switch (current_state_)
  {
    case State::INIT:
      executeInit();
      break;
    case State::IDLE:
      executeIdle();
      break;
    case State::TAKEOFF:
      executeTakeoff();
      break;
    case State::PLANNING:
      executePlanning();
      break;
    case State::FLYING:
      executeFlying();
      break;
    case State::LANDING:
      executeLanding();
      break;
    case State::LANDED:
      executeLanded();
      break;
    case State::RETURN_TO_BASE:
      executeReturnToBase();
      break;
  }
}

void StateMachine::sendOffboardSetpoint()
{
  if (current_state_ == State::IDLE || 
      current_state_ == State::TAKEOFF || 
      current_state_ == State::FLYING || 
      current_state_ == State::RETURN_TO_BASE ||
      (current_state_ == State::LANDING && !current_mode_.empty() && current_mode_ != "AUTO.LAND"))
  {
    publishSetpoint(target_position_);
  }
}

void StateMachine::transitionTo(State new_state)
{
  RCLCPP_INFO(get_logger(), "State transition: %d -> %d", 
              static_cast<int>(current_state_), static_cast<int>(new_state));
  current_state_ = new_state;
}

void StateMachine::executeInit()
{
  if (mission_.waypoints.empty())
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "No mission loaded");
    return;
  }
  
  target_position_ = current_position_;
  transitionTo(State::IDLE);
}

void StateMachine::executeIdle()
{
  target_position_ = current_position_;
  
  if (!is_armed_)
  {
    for (int i = 0; i < 10; ++i)
    {
      publishSetpoint(current_position_);
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (arm())
    {
      RCLCPP_INFO(get_logger(), "Armed");
    }
  }
  
  if (is_armed_ && !is_offboard_ && setOffboardMode())
  {
    RCLCPP_INFO(get_logger(), "Offboard mode set");
  }
  
  if (is_armed_ && is_offboard_)
  {
    transitionTo(State::TAKEOFF);
  }
}

void StateMachine::executeTakeoff()
{
  Eigen::Vector3f takeoff_position = current_position_;
  takeoff_position.z() = takeoff_altitude_;
  
  target_position_ = takeoff_position;
  
  if (isAtPosition(takeoff_position, 1.0f))
  {
    RCLCPP_INFO(get_logger(), "Takeoff complete");
    transitionTo(State::PLANNING);
  }
}

void StateMachine::executePlanning()
{
  target_position_ = current_position_;
  
  if (mission_.current_waypoint < mission_.waypoints.size())
  {
    current_goal_ = mission_.waypoints[mission_.current_waypoint];
    publishGoal(current_goal_);
  }
  else
  {
    current_goal_ = mission_.base_position;
    publishGoal(current_goal_);
    transitionTo(State::RETURN_TO_BASE);
  }
}

void StateMachine::executeFlying()
{
  if (path_index_ < current_path_.size())
  {
    target_position_ = current_path_[path_index_];
    
    if (isAtPosition(current_path_[path_index_], 0.5f))
    {
      path_index_++;
    }
  }
  else
  {
    target_position_ = current_position_;
    
    if (isAtPosition(current_goal_, 1.0f))
    {
      RCLCPP_INFO(get_logger(), "Reached waypoint %zu", mission_.current_waypoint);
      mission_.current_waypoint++;
      path_index_ = 0;
      
      if (mission_.current_waypoint < mission_.waypoints.size())
      {
        transitionTo(State::PLANNING);
      }
      else
      {
        transitionTo(State::LANDING);
      }
    }
  }
}

void StateMachine::executeLanding()
{
  if (current_mode_ == "AUTO.LAND")
  {
    transitionTo(State::LANDED);
    return;
  }
  
  if (has_landing_zone_)
  {
    Eigen::Vector3f above_landing = landing_zone_center_;
    above_landing.z() = current_position_.z();
    
    if (!isAtPosition(above_landing, 2.0f))
    {
      target_position_ = above_landing;
    }
    else
    {
      if (setLandMode())
      {
        RCLCPP_INFO(get_logger(), "Landing mode activated");
        transitionTo(State::LANDED);
      }
    }
  }
  else
  {
    target_position_ = current_position_;
    
    if (current_position_.z() < 5.0f && setLandMode())
    {
      RCLCPP_INFO(get_logger(), "Low altitude reached, switching to LAND mode");
      transitionTo(State::LANDED);
    }
  }
}

void StateMachine::executeLanded()
{
  if (!is_armed_)
  {
    RCLCPP_INFO(get_logger(), "Landed and disarmed");
    
    rclcpp::sleep_for(std::chrono::seconds(5));
    
    transitionTo(State::IDLE);
    mission_.current_waypoint = 0;
  }
}

void StateMachine::executeReturnToBase()
{
  if (path_index_ < current_path_.size())
  {
    target_position_ = current_path_[path_index_];
    
    if (isAtPosition(current_path_[path_index_], 0.5f))
    {
      path_index_++;
    }
  }
  else
  {
    target_position_ = mission_.base_position;
    
    if (isAtPosition(mission_.base_position, 1.0f))
    {
      RCLCPP_INFO(get_logger(), "Returned to base");
      transitionTo(State::LANDING);
    }
  }
}

bool StateMachine::arm()
{
  auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
  request->value = true;
  
  auto future = arming_client_->async_send_request(request);
  
  if (rclcpp::spin_until_future_complete(shared_from_this(), future, std::chrono::seconds(5)) == 
      rclcpp::FutureReturnCode::SUCCESS)
  {
    return future.get()->success;
  }
  
  RCLCPP_ERROR(get_logger(), "Arming service timeout");
  return false;
}

bool StateMachine::disarm()
{
  auto request = std::make_shared<mavros_msgs::srv::CommandBool::Request>();
  request->value = false;
  
  auto future = arming_client_->async_send_request(request);
  
  if (rclcpp::spin_until_future_complete(shared_from_this(), future, std::chrono::seconds(5)) == 
      rclcpp::FutureReturnCode::SUCCESS)
  {
    return future.get()->success;
  }
  
  RCLCPP_ERROR(get_logger(), "Disarming service timeout");
  return false;
}

bool StateMachine::setOffboardMode()
{
  auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
  request->custom_mode = "OFFBOARD";
  
  auto future = set_mode_client_->async_send_request(request);
  
  if (rclcpp::spin_until_future_complete(shared_from_this(), future, std::chrono::seconds(5)) == 
      rclcpp::FutureReturnCode::SUCCESS)
  {
    return future.get()->mode_sent;
  }
  
  RCLCPP_ERROR(get_logger(), "Set mode service timeout");
  return false;
}

bool StateMachine::setLandMode()
{
  auto request = std::make_shared<mavros_msgs::srv::SetMode::Request>();
  request->custom_mode = "AUTO.LAND";
  
  auto future = set_mode_client_->async_send_request(request);
  
  if (rclcpp::spin_until_future_complete(shared_from_this(), future, std::chrono::seconds(5)) == 
      rclcpp::FutureReturnCode::SUCCESS)
  {
    return future.get()->mode_sent;
  }
  
  RCLCPP_ERROR(get_logger(), "Set land mode service timeout");
  return false;
}

void StateMachine::publishGoal(const Eigen::Vector3f& goal)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = "map";
  msg.pose.position.x = goal.x();
  msg.pose.position.y = goal.y();
  msg.pose.position.z = goal.z();
  msg.pose.orientation.w = 1.0;
  
  goal_pub_->publish(msg);
}

void StateMachine::publishSetpoint(const Eigen::Vector3f& position)
{
  geometry_msgs::msg::PoseStamped msg;
  msg.header.stamp = now();
  msg.header.frame_id = "map";
  msg.pose.position.x = position.x();
  msg.pose.position.y = position.y();
  msg.pose.position.z = position.z();
  msg.pose.orientation.w = 1.0;
  
  setpoint_pub_->publish(msg);
}

bool StateMachine::isAtPosition(const Eigen::Vector3f& target, float tolerance)
{
  return (current_position_ - target).norm() < tolerance;
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vkr_ros::StateMachine)