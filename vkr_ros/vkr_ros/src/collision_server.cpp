#include "vkr_ros/collision_server.hpp"
#include <vkr/geometry/static_obstacles/bst_builder.hpp>
#include <vkr/terrain/qwg_builder.hpp>
#include <spdlog/spdlog.h>
#include <filesystem>

namespace vkr_ros
{

CollisionServer::CollisionServer(const rclcpp::NodeOptions& options)
  : Node("collision_server", options)
{
  spdlog::set_level(spdlog::level::info);
  
  declare_parameter("terrain_file", "");
  declare_parameter("obstacle_files", std::vector<std::string>{});
  declare_parameter("obstacle_transforms", std::vector<double>{});
  declare_parameter("wavelet_levels", 8);
  declare_parameter("page_size", 4096);
  declare_parameter("max_cache_size", 1024);
  declare_parameter("max_faces_per_leaf", 32);
  declare_parameter("grid_cell_size", 8.0);
  declare_parameter("hash_table_size", 65536);
  
  config_.terrain.wavelet_levels = get_parameter("wavelet_levels").as_int();
  config_.terrain.page_size = get_parameter("page_size").as_int();
  config_.terrain.max_cache_size = get_parameter("max_cache_size").as_int();
  config_.geometry.max_faces_per_leaf = get_parameter("max_faces_per_leaf").as_int();
  config_.geometry.grid_cell_size = get_parameter("grid_cell_size").as_double();
  config_.geometry.hash_table_size = get_parameter("hash_table_size").as_int();
  
  collision_query_ = std::make_unique<vkr::geometry::collision::CollisionQuery>(config_.geometry);
  
  std::string terrain_file = get_parameter("terrain_file").as_string();
  std::vector<std::string> obstacle_files = get_parameter("obstacle_files").as_string_array();
  std::vector<double> obstacle_transforms_flat = get_parameter("obstacle_transforms").as_double_array();
  
  if (!terrain_file.empty())
  {
    loadTerrain(terrain_file);
  }
  
  if (!obstacle_files.empty())
  {
    std::vector<std::vector<float>> obstacle_transforms;
    
    if (obstacle_transforms_flat.size() == obstacle_files.size() * 9)
    {
      for (size_t i = 0; i < obstacle_files.size(); ++i)
      {
        std::vector<float> transform(9);
        for (size_t j = 0; j < 9; ++j)
        {
          transform[j] = static_cast<float>(obstacle_transforms_flat[i * 9 + j]);
        }
        obstacle_transforms.push_back(transform);
      }
    }
    else if (obstacle_transforms_flat.empty())
    {
      for (size_t i = 0; i < obstacle_files.size(); ++i)
      {
        obstacle_transforms.push_back({0, 0, 0, 0, 0, 0, 1, 1, 1});
      }
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Invalid obstacle_transforms size. Expected %zu elements, got %zu", 
                   obstacle_files.size() * 9, obstacle_transforms_flat.size());
      return;
    }
    
    loadStaticObstacles(obstacle_files, obstacle_transforms);
  }
  
  check_capsule_srv_ = create_service<vkr_srvs::srv::CheckCapsule>(
      "/collision/check_capsule",
      std::bind(&CollisionServer::handleCheckCapsule, this, std::placeholders::_1, std::placeholders::_2));
  
  check_segment_srv_ = create_service<vkr_srvs::srv::CheckSegment>(
      "/collision/check_segment",
      std::bind(&CollisionServer::handleCheckSegment, this, std::placeholders::_1, std::placeholders::_2));
  
  check_path_srv_ = create_service<vkr_srvs::srv::CheckPath>(
      "/collision/check_path",
      std::bind(&CollisionServer::handleCheckPath, this, std::placeholders::_1, std::placeholders::_2));
  
  get_distance_srv_ = create_service<vkr_srvs::srv::GetDistance>(
      "/collision/get_distance",
      std::bind(&CollisionServer::handleGetDistance, this, std::placeholders::_1, std::placeholders::_2));
  
  RCLCPP_INFO(get_logger(), "Collision server initialized");
}

void CollisionServer::loadTerrain(const std::string& terrain_file)
{
  std::filesystem::path file_path(terrain_file);
  std::string extension = file_path.extension().string();
  std::string qwg_file;
  
  if (extension == ".tif" || extension == ".tiff")
  {
    std::string cache_dir = "/tmp/vkr_cache";
    std::filesystem::create_directories(cache_dir);
    
    qwg_file = cache_dir + "/" + file_path.stem().string() + ".qwg";
    
    if (!std::filesystem::exists(qwg_file) || 
        std::filesystem::last_write_time(terrain_file) > std::filesystem::last_write_time(qwg_file))
    {
      RCLCPP_INFO(get_logger(), "Converting GeoTIFF to QWG format...");
      convertGeoTIFFToQWG(terrain_file, qwg_file);
    }
    else
    {
      RCLCPP_INFO(get_logger(), "Using cached QWG file: %s", qwg_file.c_str());
    }
  }
  else if (extension == ".qwg")
  {
    qwg_file = terrain_file;
  }
  else
  {
    RCLCPP_ERROR(get_logger(), "Unsupported terrain file format: %s", extension.c_str());
    return;
  }
  
  terrain_ = std::make_shared<vkr::terrain::WaveletGrid>(config_.terrain);
  
  if (terrain_->loadFromFile(qwg_file))
  {
    collision_query_->setTerrain(terrain_);
    RCLCPP_INFO(get_logger(), "Loaded terrain from %s", qwg_file.c_str());
  }
  else
  {
    RCLCPP_ERROR(get_logger(), "Failed to load terrain from %s", qwg_file.c_str());
  }
}

void CollisionServer::convertGeoTIFFToQWG(const std::string& geotiff_file, const std::string& qwg_file)
{
  vkr::terrain::QWGBuilder builder(config_.terrain);
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(
      geotiff_file, 
      qwg_file, 
      error_message,
      [this](size_t current, size_t total, const std::string& status)
      {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
                             "QWG conversion: %zu/%zu - %s", 
                             current, total, status.c_str());
      });
  
  if (success)
  {
    RCLCPP_INFO(get_logger(), "Successfully converted GeoTIFF to QWG");
  }
  else
  {
    RCLCPP_ERROR(get_logger(), "Failed to convert GeoTIFF: %s", error_message.c_str());
  }
}

void CollisionServer::loadStaticObstacles(const std::vector<std::string>& obstacle_files,
                                         const std::vector<std::vector<float>>& obstacle_transforms)
{
  auto builder = std::make_unique<vkr::geometry::static_obstacles::BSTBuilder>(config_.geometry);
  
  for (size_t i = 0; i < obstacle_files.size(); ++i)
  {
    const auto& file = obstacle_files[i];
    const auto& transform = obstacle_transforms[i];
    
    Eigen::Vector3f position(transform[0], transform[1], transform[2]);
    Eigen::Vector3f rotation(transform[3], transform[4], transform[5]);
    Eigen::Vector3f scale(transform[6], transform[7], transform[8]);
    
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    
    Eigen::AngleAxisf rollAngle(rotation.x(), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(rotation.y(), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(rotation.z(), Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = yawAngle * pitchAngle * rollAngle;
    
    transform_matrix.block<3, 3>(0, 0) = q.matrix() * Eigen::Matrix3f(scale.asDiagonal());
    transform_matrix.block<3, 1>(0, 3) = position;
    
    if (builder->loadMesh(file, i + 1))
    {
      RCLCPP_INFO(get_logger(), "Loaded obstacle %zu from %s at [%.2f, %.2f, %.2f]", 
                  i + 1, file.c_str(), position.x(), position.y(), position.z());
      
      builder->transformLastMesh(transform_matrix);
    }
    else
    {
      RCLCPP_ERROR(get_logger(), "Failed to load obstacle from %s", file.c_str());
    }
  }
  
  if (builder->getFaceCount() > 0)
  {
    static_obstacles_ = builder->build();
    collision_query_->setStaticObstacles(static_obstacles_);
    RCLCPP_INFO(get_logger(), "Built BST with %zu faces from %zu files", 
                builder->getFaceCount(), obstacle_files.size());
  }
}

void CollisionServer::handleCheckCapsule(
    const std::shared_ptr<vkr_srvs::srv::CheckCapsule::Request> request,
    std::shared_ptr<vkr_srvs::srv::CheckCapsule::Response> response)
{
  std::lock_guard<std::mutex> lock(collision_mutex_);
  
  vkr::geometry::Capsule capsule;
  capsule.p0 = Eigen::Vector3f(request->p0.x, request->p0.y, request->p0.z);
  capsule.p1 = Eigen::Vector3f(request->p1.x, request->p1.y, request->p1.z);
  capsule.radius = request->radius;
  
  auto time = rosTimeToSteady(request->check_time);
  
  auto result = collision_query_->queryCapsule(capsule, time, request->uav_id);
  
  response->collision_detected = result.hit;
  response->object_id = result.object_id;
  response->object_type = objectTypeToString(result.object_type);
  response->hit_point.x = result.hit_point.x();
  response->hit_point.y = result.hit_point.y();
  response->hit_point.z = result.hit_point.z();
  response->hit_normal.x = result.hit_normal.x();
  response->hit_normal.y = result.hit_normal.y();
  response->hit_normal.z = result.hit_normal.z();
}

void CollisionServer::handleCheckSegment(
    const std::shared_ptr<vkr_srvs::srv::CheckSegment::Request> request,
    std::shared_ptr<vkr_srvs::srv::CheckSegment::Response> response)
{
  std::lock_guard<std::mutex> lock(collision_mutex_);
  
  vkr::geometry::LineSegment segment;
  segment.start = Eigen::Vector3f(request->start.x, request->start.y, request->start.z);
  segment.end = Eigen::Vector3f(request->end.x, request->end.y, request->end.z);
  
  auto time = rosTimeToSteady(request->check_time);
  
  auto result = collision_query_->querySegment(segment, time, request->uav_id);
  
  response->collision_detected = result.hit;
  response->object_id = result.object_id;
  response->object_type = objectTypeToString(result.object_type);
  response->hit_point.x = result.hit_point.x();
  response->hit_point.y = result.hit_point.y();
  response->hit_point.z = result.hit_point.z();
  response->hit_normal.x = result.hit_normal.x();
  response->hit_normal.y = result.hit_normal.y();
  response->hit_normal.z = result.hit_normal.z();
}

void CollisionServer::handleCheckPath(
    const std::shared_ptr<vkr_srvs::srv::CheckPath::Request> request,
    std::shared_ptr<vkr_srvs::srv::CheckPath::Response> response)
{
  std::lock_guard<std::mutex> lock(collision_mutex_);
  
  std::vector<Eigen::Vector3f> path;
  path.reserve(request->waypoints.size());
  
  for (const auto& wp : request->waypoints)
  {
    path.emplace_back(wp.x, wp.y, wp.z);
  }
  
  auto time = rosTimeToSteady(request->start_time);
  
  auto result = collision_query_->queryPath(path, request->radius, time, request->velocity, request->uav_id);
  
  response->collision_detected = result.hit;
  response->object_id = result.object_id;
  response->object_type = objectTypeToString(result.object_type);
  response->hit_point.x = result.hit_point.x();
  response->hit_point.y = result.hit_point.y();
  response->hit_point.z = result.hit_point.z();
  
  if (result.hit)
  {
    float distance_traveled = 0.0f;
    for (size_t i = 1; i < path.size(); ++i)
    {
      float segment_length = (path[i] - path[i-1]).norm();
      if (distance_traveled + segment_length >= result.t_min)
      {
        response->first_collision_segment = i - 1;
        break;
      }
      distance_traveled += segment_length;
    }
  }
}

void CollisionServer::handleGetDistance(
    const std::shared_ptr<vkr_srvs::srv::GetDistance::Request> request,
    std::shared_ptr<vkr_srvs::srv::GetDistance::Response> response)
{
  std::lock_guard<std::mutex> lock(collision_mutex_);
  
  Eigen::Vector3f position(request->position.x, request->position.y, request->position.z);
  auto time = rosTimeToSteady(request->check_time);
  
  response->distance = collision_query_->getDistance(position, time, request->uav_id);
  
  auto obstacles = collision_query_->getObstaclesInRadius(position, response->distance + 1.0f, time);
  
  if (!obstacles.empty())
  {
    const auto& closest = obstacles[0];
    response->closest_object_id = closest.id;
    response->closest_object_type = objectTypeToString(closest.type);
    response->closest_point.x = closest.closest_point.x();
    response->closest_point.y = closest.closest_point.y();
    response->closest_point.z = closest.closest_point.z();
  }
}

std::chrono::steady_clock::time_point CollisionServer::rosTimeToSteady(const builtin_interfaces::msg::Time& ros_time) const
{
  auto duration_ns = ros_time.sec * 1000000000LL + ros_time.nanosec;
  return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(duration_ns));
}

std::string CollisionServer::objectTypeToString(vkr::geometry::collision::HitResult::ObjectType type) const
{
  switch (type)
  {
    case vkr::geometry::collision::HitResult::ObjectType::STATIC_OBSTACLE:
      return "STATIC_OBSTACLE";
    case vkr::geometry::collision::HitResult::ObjectType::CORRIDOR:
      return "CORRIDOR";
    case vkr::geometry::collision::HitResult::ObjectType::TERRAIN:
      return "TERRAIN";
    default:
      return "NONE";
  }
}

}

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(vkr_ros::CollisionServer)