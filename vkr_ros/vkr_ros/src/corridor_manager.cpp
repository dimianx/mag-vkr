#include "vkr_ros/corridor_manager.hpp"
#include <spdlog/spdlog.h>
#include <tbb/concurrent_queue.h>

namespace vkr_ros
{

CorridorManager::CorridorManager(const rclcpp::NodeOptions& options)
  : Node("corridor_manager", options)
{
  declare_parameter("max_uavs", 10);
  declare_parameter("corridor_radius", 2.5);
  declare_parameter("corridor_coverage_fraction", 0.8);
  declare_parameter("corridor_free_distance", 5.0);
  declare_parameter("grid_cell_size", 8.0);
  declare_parameter("hash_table_size", 65536);
  
  corridor_free_distance_ = get_parameter("corridor_free_distance").as_double();
  
  config_.geometry.corridor_radius = get_parameter("corridor_radius").as_double();
  config_.geometry.corridor_coverage_fraction = get_parameter("corridor_coverage_fraction").as_double();
  config_.geometry.grid_cell_size = get_parameter("grid_cell_size").as_double();
  config_.geometry.hash_table_size = get_parameter("hash_table_size").as_int();
  
  spatial_map_ = std::make_shared<vkr::geometry::corridors::SpatialCorridorMap>(config_.geometry);
  
  free_event_pub_ = create_publisher<vkr_msgs::msg::CorridorFreeEvent>("/corridors/free_events", 10);
  
  create_corridor_srv_ = create_service<vkr_srvs::srv::CreateCorridor>(
      "/corridors/create",
      std::bind(&CorridorManager::handleCreateCorridor, this, std::placeholders::_1, std::placeholders::_2));
  
  update_corridor_srv_ = create_service<vkr_srvs::srv::UpdateCorridor>(
      "/corridors/update",
      std::bind(&CorridorManager::handleUpdateCorridor, this, std::placeholders::_1, std::placeholders::_2));
  
  remove_corridor_srv_ = create_service<vkr_srvs::srv::RemoveCorridor>(
      "/corridors/remove",
      std::bind(&CorridorManager::handleRemoveCorridor, this, std::placeholders::_1, std::placeholders::_2));
  
  check_intrusion_srv_ = create_service<vkr_srvs::srv::CheckIntrusion>(
      "/corridors/check_intrusion",
      std::bind(&CorridorManager::handleCheckIntrusion, this, std::placeholders::_1, std::placeholders::_2));
  
  for (int i = 1; i <= max_uavs_; ++i)
  {
    auto topic = "/uav_" + std::to_string(i) + "/mavros/local_position/pose";
    position_subs_[i] = create_subscription<geometry_msgs::msg::PoseStamped>(
        topic, 10,
        [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg)
        {
          positionCallback(msg, i);
        });
  }
  
  check_timer_ = create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&CorridorManager::checkAndFreeSegments, this));
  
  publish_timer_ = create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&CorridorManager::publishQueuedEvents, this));
  
  RCLCPP_INFO(get_logger(), "Corridor manager initialized for %d UAVs", max_uavs_);
}

void CorridorManager::handleCreateCorridor(
    const std::shared_ptr<vkr_srvs::srv::CreateCorridor::Request> request,
    std::shared_ptr<vkr_srvs::srv::CreateCorridor::Response> response)
{
  std::vector<Eigen::Vector3f> path;
  path.reserve(request->path.size());
  
  for (const auto& p : request->path)
  {
    path.emplace_back(p.x, p.y, p.z);
  }
  
  std::unique_lock lock(corridor_operations_mutex_);
  
  auto corridor_id = spatial_map_->createCorridor(
      request->uav_id, path, request->radius, request->current_waypoint_index);
  
  if (corridor_id != vkr::INVALID_ID)
  {
    response->success = true;
    response->corridor_id = corridor_id;
    
    std::lock_guard<std::mutex> state_lock(states_mutex_);
    uav_states_[request->uav_id].active_corridors.push_back(corridor_id);
    
    RCLCPP_INFO(get_logger(), "Created corridor %u for UAV %u", corridor_id, request->uav_id);
  }
  else
  {
    response->success = false;
    response->error_message = "Failed to create corridor";
  }
}

void CorridorManager::handleUpdateCorridor(
    const std::shared_ptr<vkr_srvs::srv::UpdateCorridor::Request> request,
    std::shared_ptr<vkr_srvs::srv::UpdateCorridor::Response> response)
{
  std::vector<Eigen::Vector3f> path;
  path.reserve(request->full_path.size());
  
  for (const auto& p : request->full_path)
  {
    path.emplace_back(p.x, p.y, p.z);
  }
  
  std::unique_lock lock(corridor_operations_mutex_);
  
  spatial_map_->updateCorridorCoverage(request->corridor_id, path, request->new_waypoint_index);
  
  response->success = true;
  RCLCPP_DEBUG(get_logger(), "Updated corridor %u coverage", request->corridor_id);
}

void CorridorManager::handleRemoveCorridor(
    const std::shared_ptr<vkr_srvs::srv::RemoveCorridor::Request> request,
    std::shared_ptr<vkr_srvs::srv::RemoveCorridor::Response> response)
{
  std::unique_lock lock(corridor_operations_mutex_);
  
  spatial_map_->removeCorridor(request->corridor_id);
  
  lock.unlock();
  
  std::lock_guard<std::mutex> state_lock(states_mutex_);
  for (auto& [uav_id, state] : uav_states_)
  {
    auto it = std::find(state.active_corridors.begin(), state.active_corridors.end(), request->corridor_id);
    if (it != state.active_corridors.end())
    {
      state.active_corridors.erase(it);
      break;
    }
  }
  
  response->success = true;
  RCLCPP_INFO(get_logger(), "Removed corridor %u", request->corridor_id);
}

void CorridorManager::handleCheckIntrusion(
    const std::shared_ptr<vkr_srvs::srv::CheckIntrusion::Request> request,
    std::shared_ptr<vkr_srvs::srv::CheckIntrusion::Response> response)
{
  Eigen::Vector3f position(request->position.x, request->position.y, request->position.z);
  
  std::shared_lock lock(corridor_operations_mutex_);
  
  response->inside_other_corridor = spatial_map_->isInsideOtherCorridor(position, request->uav_id);
  response->distance_to_boundary = spatial_map_->getDistanceToBoundary(position, request->uav_id);
  
  if (response->inside_other_corridor)
  {
    response->intruded_corridor_id = 0;
  }
}

void CorridorManager::positionCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg, vkr::UAVId uav_id)
{
  std::lock_guard<std::mutex> lock(states_mutex_);
  
  auto& state = uav_states_[uav_id];
  state.id = uav_id;
  state.position = Eigen::Vector3f(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
  state.last_update = now();
}

void CorridorManager::checkAndFreeSegments()
{
  std::vector<std::pair<vkr::CorridorId, std::vector<vkr::SegmentId>>> segments_to_free;
  
  {
    std::lock_guard<std::mutex> state_lock(states_mutex_);
    std::shared_lock spatial_lock(corridor_operations_mutex_);
    
    for (const auto& [uav_id, state] : uav_states_)
    {
      for (auto corridor_id : state.active_corridors)
      {
        vkr::geometry::corridors::Corridor corridor_info;
        if (spatial_map_->getCorridorInfo(corridor_id, corridor_info))
        {
          segments_to_free.emplace_back(
            corridor_id, 
            std::vector<vkr::SegmentId>(
              corridor_info.active_segments.begin(),
              corridor_info.active_segments.end()
            )
          );
        }
      }
    }
  }
  
  {
    std::unique_lock lock(corridor_operations_mutex_);
    
    for (const auto& [corridor_id, segments] : segments_to_free)
    {
      for (const auto& segment_id : segments)
      {
        spatial_map_->freeSegment(corridor_id, segment_id);
      }
    }
    
    vkr::geometry::corridors::CorridorFreeEvent event;
    while (spatial_map_->getNextFreeEvent(event))
    {
      vkr_msgs::msg::CorridorFreeEvent msg;
      msg.corridor_id = event.corridor_id;
      msg.segment_id = event.segment_id;
      msg.timestamp.sec = std::chrono::duration_cast<std::chrono::seconds>(
          event.timestamp.time_since_epoch()).count();
      msg.timestamp.nanosec = std::chrono::duration_cast<std::chrono::nanoseconds>(
          event.timestamp.time_since_epoch()).count() % 1000000000;
      
      vkr::geometry::corridors::Corridor corridor_info;
      if (spatial_map_->getCorridorInfo(event.corridor_id, corridor_info))
      {
        float segment_length = 10.0f;
        msg.min_bound.x = -segment_length;
        msg.min_bound.y = -segment_length;
        msg.min_bound.z = -segment_length;
        msg.max_bound.x = segment_length;
        msg.max_bound.y = segment_length;
        msg.max_bound.z = segment_length;
        msg.uav_id = corridor_info.owner;
      }
      
      event_queue_.push(msg);
    }
  }
}

void CorridorManager::publishQueuedEvents()
{
  vkr_msgs::msg::CorridorFreeEvent msg;
  
  while (event_queue_.try_pop(msg))
  {
    free_event_pub_->publish(msg);
  }
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vkr_ros::CorridorManager)