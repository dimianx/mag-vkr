#include "vkr_ros/planner_node.hpp"
#include <spdlog/spdlog.h>

namespace vkr_ros
{

static std::mutex global_planning_mutex;
static std::condition_variable planning_cv;
static int current_planner_id = 0;
static std::chrono::steady_clock::time_point last_planning_time;

PlannerNode::PlannerNode(const rclcpp::NodeOptions& options)
  : Node("planner_node", options),
    has_goal_(false),
    has_path_(false),
    planning_in_progress_(false),
    needs_replan_(false)
{
  declare_parameter("uav_id", 1);
  declare_parameter("coarse_grid_size", 4.0);
  declare_parameter("fine_grid_size", 1.0);
  declare_parameter("initial_epsilon", 2.5);
  declare_parameter("corridor_penalty_weight", 1000.0);
  declare_parameter("height_penalty_weight", 10.0);
  declare_parameter("max_runtime_ms", 100.0);
  declare_parameter("safe_altitude", 20.0);
  declare_parameter("min_altitude", 10.0);
  declare_parameter("min_clearance", 5.0);
  declare_parameter("planning_timeout_ms", 5000);
  
  uav_id_ = get_parameter("uav_id").as_int();
  planning_timeout_ms_ = get_parameter("planning_timeout_ms").as_int();
  
  config_.planning.coarse_grid_size = get_parameter("coarse_grid_size").as_double();
  config_.planning.fine_grid_size = get_parameter("fine_grid_size").as_double();
  config_.planning.initial_epsilon = get_parameter("initial_epsilon").as_double();
  config_.planning.corridor_penalty_weight = get_parameter("corridor_penalty_weight").as_double();
  config_.planning.height_penalty_weight = get_parameter("height_penalty_weight").as_double();
  config_.planning.max_runtime_ms = get_parameter("max_runtime_ms").as_double();
  config_.planning.safe_altitude = get_parameter("safe_altitude").as_double();
  config_.planning.min_altitude = get_parameter("min_altitude").as_double();
  config_.planning.min_clearance = get_parameter("min_clearance").as_double();
  
  planner_ = std::make_unique<vkr::planning::AMCPPlanner>(config_);
  
  std::string ns = get_namespace();
  
  goal_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ns + "/goal_pose", 10,
      std::bind(&PlannerNode::goalCallback, this, std::placeholders::_1));
  
  position_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      ns + "/mavros/local_position/pose", 10,
      std::bind(&PlannerNode::positionCallback, this, std::placeholders::_1));
  
  corridor_free_sub_ = create_subscription<vkr_msgs::msg::CorridorFreeEvent>(
      "/corridors/free_events", 10,
      std::bind(&PlannerNode::corridorFreeCallback, this, std::placeholders::_1));
  
  path_pub_ = create_publisher<vkr_msgs::msg::Path>(ns + "/planned_path", 10);
  
  check_capsule_client_ = create_client<vkr_srvs::srv::CheckCapsule>("/collision/check_capsule");
  create_corridor_client_ = create_client<vkr_srvs::srv::CreateCorridor>("/corridors/create");
  
  replan_timer_ = create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&PlannerNode::checkForReplan, this));
  
  RCLCPP_INFO(get_logger(), "Planner node initialized for UAV %d", uav_id_);
}

void PlannerNode::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  goal_position_ = Eigen::Vector3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  has_goal_ = true;
  
  RCLCPP_INFO(get_logger(), "Received new goal: [%.2f, %.2f, %.2f]", 
              goal_position_.x(), goal_position_.y(), goal_position_.z());
  
  if (!planning_in_progress_)
  {
    planPath();
  }
}

void PlannerNode::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  current_position_ = Eigen::Vector3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
}

void PlannerNode::corridorFreeCallback(const vkr_msgs::msg::CorridorFreeEvent::SharedPtr msg)
{
  vkr::geometry::corridors::CorridorFreeEvent event;
  event.corridor_id = msg->corridor_id;
  event.segment_id = msg->segment_id;
  event.timestamp = std::chrono::steady_clock::time_point(
      std::chrono::seconds(msg->timestamp.sec) + 
      std::chrono::nanoseconds(msg->timestamp.nanosec));
  
  planner_->pushEvent(event);
  
  if (has_path_ && !planning_in_progress_)
  {
    needs_replan_ = true;
    RCLCPP_DEBUG(get_logger(), "Corridor segment freed, marking for replan");
  }
}

bool PlannerNode::acquirePlanningLock()
{
  std::unique_lock<std::mutex> lock(global_planning_mutex);
  
  auto start_time = std::chrono::steady_clock::now();
  auto timeout = std::chrono::milliseconds(planning_timeout_ms_);
  
  while (current_planner_id != 0 && current_planner_id != uav_id_)
  {
    auto now = std::chrono::steady_clock::now();
    
    if (now - last_planning_time > std::chrono::seconds(10))
    {
      RCLCPP_WARN(get_logger(), "Planning lock timeout, forcefully acquiring for UAV %d", uav_id_);
      break;
    }
    
    if (now - start_time > timeout)
    {
      RCLCPP_ERROR(get_logger(), "Failed to acquire planning lock within timeout");
      return false;
    }
    
    RCLCPP_INFO(get_logger(), "UAV %d waiting for planning lock (held by UAV %d)", 
                uav_id_, current_planner_id);
    
    planning_cv.wait_for(lock, std::chrono::milliseconds(100));
  }
  
  current_planner_id = uav_id_;
  last_planning_time = std::chrono::steady_clock::now();
  
  RCLCPP_INFO(get_logger(), "UAV %d acquired planning lock", uav_id_);
  return true;
}

void PlannerNode::releasePlanningLock()
{
  std::unique_lock<std::mutex> lock(global_planning_mutex);
  
  if (current_planner_id == uav_id_)
  {
    current_planner_id = 0;
    RCLCPP_INFO(get_logger(), "UAV %d released planning lock", uav_id_);
    planning_cv.notify_all();
  }
}

void PlannerNode::planPath()
{
  if (!has_goal_)
  {
    return;
  }
  
  if (!acquirePlanningLock())
  {
    RCLCPP_ERROR(get_logger(), "Failed to acquire planning lock, aborting planning");
    return;
  }
  
  planning_in_progress_ = true;
  needs_replan_ = false;
  
  vkr::planning::PlanRequest request;
  request.start = current_position_;
  request.goal = goal_position_;
  request.uav_id = uav_id_;
  request.max_runtime_ms = config_.planning.max_runtime_ms;
  
  planning_future_ = planner_->planAsync(request);
  
  auto timer = create_wall_timer(
      std::chrono::milliseconds(10),
      [this]()
      {
        if (planning_future_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
          auto result = planning_future_.get();
          
          if (result.isSuccess())
          {
            current_plan_ = result;
            has_path_ = true;
            publishPath(result);
          }
          else
          {
            RCLCPP_WARN(get_logger(), "Planning failed");
            has_path_ = false;
          }
          
          planning_in_progress_ = false;
          releasePlanningLock();
          
          if (!has_path_)
          {
            has_goal_ = false;
          }
        }
      });
}

void PlannerNode::checkForReplan()
{
  if (needs_replan_ && has_path_ && has_goal_ && !planning_in_progress_)
  {
    RCLCPP_INFO(get_logger(), "Replanning due to corridor changes");
    planPath();
  }
}

void PlannerNode::publishPath(const vkr::planning::PlanResult& result)
{
  vkr_msgs::msg::Path path_msg;
  path_msg.header.stamp = now();
  path_msg.header.frame_id = "map";
  
  for (const auto& wp : result.path)
  {
    geometry_msgs::msg::Point p;
    p.x = wp.x();
    p.y = wp.y();
    p.z = wp.z();
    path_msg.waypoints.push_back(p);
  }
  
  path_msg.total_cost = result.cost;
  path_msg.epsilon = result.epsilon;
  path_msg.nodes_expanded = result.nodes_expanded;
  path_msg.nodes_generated = result.nodes_generated;
  path_msg.planning_time_ms = result.planning_time.count();
  path_msg.path_length = result.path_length;
  path_msg.min_clearance = result.min_clearance;
  path_msg.max_altitude = result.max_altitude;
  
  path_pub_->publish(path_msg);
  
  RCLCPP_INFO(get_logger(), "Published path with %zu waypoints, cost: %.2f", 
              result.path.size(), result.cost);
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vkr_ros::PlannerNode)