#include "vkr/geometry/collision/collision_query.hpp"
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/geometry/dynamic_obstacles/swept_obb_bvh.hpp"
#include "vkr/geometry/dynamic_obstacles/swept_obb.hpp"
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/geometry/intersections.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <unordered_map>

namespace vkr
{
namespace geometry
{
namespace collision
{

namespace
{

dynamic_obstacles::OBB capsuleToOBB(const Capsule& capsule) 
{
  dynamic_obstacles::OBB obb;
  
  obb.center = (capsule.p0 + capsule.p1) * 0.5f;
  
  Eigen::Vector3f axis = capsule.p1 - capsule.p0;
  float length = axis.norm();
  
  if (length > GEOM_EPS) 
  {
    axis.normalize();
    
    Eigen::Vector3f up(0, 0, 1);
    if (std::abs(axis.dot(up)) > 0.99f) 
    {
      up = Eigen::Vector3f(1, 0, 0);
    }
    
    Eigen::Vector3f right = axis.cross(up).normalized();
    Eigen::Vector3f forward = right.cross(axis).normalized();
    
    obb.orientation.col(0) = axis;
    obb.orientation.col(1) = right;
    obb.orientation.col(2) = forward;
  }
  else 
  {
    obb.orientation = Eigen::Matrix3f::Identity();
  }
  
  obb.half_extents = Eigen::Vector3f(
    length * 0.5f + capsule.radius,
    capsule.radius,
    capsule.radius
  );
  
  return obb;
}

float pointToOBBDistance(const Eigen::Vector3f& point, const dynamic_obstacles::OBB& obb) 
{
  Eigen::Vector3f local_point = obb.orientation.transpose() * (point - obb.center);
  
  Eigen::Vector3f closest_local;
  for (int i = 0; i < 3; ++i) 
  {
    closest_local[i] = std::clamp(local_point[i], -obb.half_extents[i], obb.half_extents[i]);
  }
  
  Eigen::Vector3f closest_world = obb.center + obb.orientation * closest_local;
  return (point - closest_world).norm();
}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}

struct CollisionQuery::Implementation 
{
  GeometryConfig config;
  
  std::shared_ptr<terrain::WaveletGrid> terrain;
  std::shared_ptr<static_obstacles::BoundingSphereTree> static_obstacles;
  std::shared_ptr<dynamic_obstacles::SweptOBBBVH> dynamic_obstacles;
  std::shared_ptr<corridors::SpatialCorridorMap> corridors;
  
  std::unique_ptr<UniformGrid> static_grid;
  
  bool early_exit_enabled = true;
  bool cache_enabled = true;
  
  struct CacheEntry 
  {
    HitResult result;
    std::chrono::steady_clock::time_point timestamp;
  };
  mutable std::unordered_map<size_t, CacheEntry> query_cache;
  mutable std::mutex cache_mutex;
  mutable std::chrono::steady_clock::time_point last_cache_cleanup;
  static constexpr auto CACHE_DURATION = std::chrono::milliseconds(100);
  static constexpr size_t MAX_CACHE_SIZE = 10000;
  
  mutable PerformanceStats stats;
  mutable std::mutex stats_mutex;
  mutable std::chrono::steady_clock::time_point last_query_start;
  
  Implementation(const GeometryConfig& cfg) 
    : config(cfg)
    , last_cache_cleanup(std::chrono::steady_clock::now()) 
  {
    spdlog::debug("CollisionQuery initialized with grid_cell_size={}, hash_table_size={}", 
                  config.grid_cell_size, config.hash_table_size);
  }
  
  void initializeStaticGrid() 
  {
    if (!static_obstacles) return;
    
    auto stats = static_obstacles->getStats();
    if (stats.total_nodes == 0) return;
    
    BoundingBox world_bounds;
    if (terrain) 
    {
      world_bounds = terrain->getBounds();
      world_bounds.min.z() -= 100.0f;
      world_bounds.max.z() += 500.0f;
    }
    else 
    {
      world_bounds.min = Eigen::Vector3f(-1000, -1000, -100);
      world_bounds.max = Eigen::Vector3f(1000, 1000, 500);
    }
    
    static_grid = std::make_unique<UniformGrid>(
      config.grid_cell_size,
      world_bounds,
      config.hash_table_size
    );
    
    static_grid->insert(0, world_bounds);
    
    spdlog::info("Initialized static obstacle spatial grid:");
    spdlog::info("  - World bounds: [{:.1f}, {:.1f}, {:.1f}] to [{:.1f}, {:.1f}, {:.1f}]",
                 world_bounds.min.x(), world_bounds.min.y(), world_bounds.min.z(),
                 world_bounds.max.x(), world_bounds.max.y(), world_bounds.max.z());
    spdlog::info("  - Grid cell size: {:.1f}m", config.grid_cell_size);
    spdlog::info("  - Hash table size: {}", config.hash_table_size);
    spdlog::debug("  - BST stats: {} nodes, {} leaves, depth {}", 
                  stats.total_nodes, stats.leaf_nodes, stats.max_depth);
  }
  
  HitResult checkTerrain(const Eigen::Vector3f& position, float radius) const 
  {
    HitResult result;
    if (!terrain) return result;
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.terrain_checks++;
    }
    
    auto terrain_bounds = terrain->getBounds();
    if (position.x() < terrain_bounds.min.x() || position.x() > terrain_bounds.max.x() ||
        position.y() < terrain_bounds.min.y() || position.y() > terrain_bounds.max.y()) 
    {
      return result;
    }
    
    std::vector<Eigen::Vector3f> check_points;
    check_points.push_back(position);
    check_points.push_back(position + Eigen::Vector3f(radius, 0, 0));
    check_points.push_back(position + Eigen::Vector3f(-radius, 0, 0));
    check_points.push_back(position + Eigen::Vector3f(0, radius, 0));
    check_points.push_back(position + Eigen::Vector3f(0, -radius, 0));
    
    float min_clearance = std::numeric_limits<float>::max();
    Eigen::Vector3f closest_terrain_point = position;
    bool found_valid_point = false;
    
    for (const auto& point : check_points) 
    {
      if (point.x() < terrain_bounds.min.x() || point.x() > terrain_bounds.max.x() ||
          point.y() < terrain_bounds.min.y() || point.y() > terrain_bounds.max.y()) 
      {
        continue;
      }
      
      float terrain_height = terrain->getHeight(point.x(), point.y());
      float clearance = point.z() - terrain_height;
      
      if (clearance < min_clearance) 
      {
        min_clearance = clearance;
        closest_terrain_point = Eigen::Vector3f(point.x(), point.y(), terrain_height);
        found_valid_point = true;
      }
    }
    
    if (!found_valid_point) 
    {
      return result;
    }
    
    if (min_clearance <= radius) 
    {
      result.hit = true;
      result.t_min = 0.0f;
      result.object_type = HitResult::ObjectType::TERRAIN;
      result.hit_point = closest_terrain_point;
      
      float h = 1.0f;
      
      float x_minus = std::max(closest_terrain_point.x() - h, terrain_bounds.min.x());
      float x_plus = std::min(closest_terrain_point.x() + h, terrain_bounds.max.x());
      float y_minus = std::max(closest_terrain_point.y() - h, terrain_bounds.min.y());
      float y_plus = std::min(closest_terrain_point.y() + h, terrain_bounds.max.y());
      
      float h_x0 = terrain->getHeight(x_minus, closest_terrain_point.y());
      float h_x1 = terrain->getHeight(x_plus, closest_terrain_point.y());
      float h_y0 = terrain->getHeight(closest_terrain_point.x(), y_minus);
      float h_y1 = terrain->getHeight(closest_terrain_point.x(), y_plus);
      
      float dx = x_plus - x_minus;
      float dy = y_plus - y_minus;
      
      Eigen::Vector3f normal(
        -(h_x1 - h_x0) / dx,
        -(h_y1 - h_y0) / dy,
        1.0f
      );
      result.hit_normal = normal.normalized();
    }
    
    return result;
  }
  
  HitResult checkStaticObstacles(const Capsule& capsule) const 
  {
    HitResult result;
    if (!static_obstacles) return result;
    
    if (static_grid) 
    {
      BoundingBox capsule_bbox;
      capsule_bbox.min = capsule.p0.cwiseMin(capsule.p1) - Eigen::Vector3f::Constant(capsule.radius);
      capsule_bbox.max = capsule.p0.cwiseMax(capsule.p1) + Eigen::Vector3f::Constant(capsule.radius);
      
      auto candidates = static_grid->query(capsule_bbox);
      
      {
        std::lock_guard<std::mutex> lock(stats_mutex);
        stats.broad_phase_checks += candidates.size();
      }
      
      if (candidates.empty()) 
      {
        return result;
      }
    }
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.narrow_phase_checks++;
      stats.static_checks++;
    }
    
    static_obstacles->intersectCapsule(capsule, result);
    
    if (result.hit) 
    {
      result.object_type = HitResult::ObjectType::STATIC_OBSTACLE;
    }
    
    return result;
  }
  
  HitResult checkDynamicObstacles(const Capsule& capsule,
                                 const std::chrono::steady_clock::time_point& t0,
                                 const std::chrono::steady_clock::time_point& t1) const 
  {
    HitResult result;
    if (!dynamic_obstacles) return result;
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.dynamic_checks++;
    }
    
    dynamic_obstacles::OBB obb = capsuleToOBB(capsule);
    
    auto dynamic_result = dynamic_obstacles->queryOBB(obb, t0, t1);
    
    if (dynamic_result.hit) 
    {
      result = dynamic_result;
      result.object_type = HitResult::ObjectType::DYNAMIC_OBSTACLE;
    }
    
    return result;
  }
  
  HitResult checkCorridors(const Capsule& capsule, UAVId uav_id) const 
  {
    HitResult result;
    if (!corridors) return result;
    
    {
      std::lock_guard<std::mutex> lock(stats_mutex);
      stats.corridor_checks++;
    }
    
    if (corridors->intersectCapsule(capsule, uav_id)) 
    {
      result.hit = true;
      result.t_min = 0.0f;
      result.object_type = HitResult::ObjectType::CORRIDOR;
      
      Eigen::Vector3f direction = capsule.p1 - capsule.p0;
      float length = direction.norm();
      if (length > GEOM_EPS) 
      {
        direction.normalize();
        
        const int num_samples = 10;
        for (int i = 0; i <= num_samples; ++i) 
        {
          float t = static_cast<float>(i) / num_samples;
          Eigen::Vector3f point = capsule.p0 + t * length * direction;
          
          if (corridors->isInsideOtherCorridor(point, uav_id)) 
          {
            result.hit_point = point;
            
            float delta = 1.0f;
            Eigen::Vector3f gradient(0, 0, 0);
            
            for (int axis = 0; axis < 3; ++axis) 
            {
              Eigen::Vector3f offset = Eigen::Vector3f::Zero();
              offset[axis] = delta;
              
              bool inside_plus = corridors->isInsideOtherCorridor(point + offset, uav_id);
              bool inside_minus = corridors->isInsideOtherCorridor(point - offset, uav_id);
              
              if (inside_plus && !inside_minus) 
              {
                gradient[axis] = -1.0f;
              }
              else if (!inside_plus && inside_minus) 
              {
                gradient[axis] = 1.0f;
              }
            }
            
            if (gradient.norm() > GEOM_EPS) 
            {
              result.hit_normal = gradient.normalized();
            }
            else 
            {
              Eigen::Vector3f capsule_center = (capsule.p0 + capsule.p1) * 0.5f;
              result.hit_normal = (capsule_center - point).normalized();
            }
            
            break;
          }
        }
      }
      else 
      {
        result.hit_point = capsule.p0;
        result.hit_normal = Eigen::Vector3f(0, 0, 1);
      }
    }
    
    return result;
  }
  
  size_t computeCacheKey(const Capsule& capsule) const 
  {
    size_t seed = 0;
    hash_combine(seed, capsule.p0.x());
    hash_combine(seed, capsule.p0.y());
    hash_combine(seed, capsule.p0.z());
    hash_combine(seed, capsule.p1.x());
    hash_combine(seed, capsule.p1.y());
    hash_combine(seed, capsule.p1.z());
    hash_combine(seed, capsule.radius);
    return seed;
  }
  
  void cleanupCache(const std::chrono::steady_clock::time_point& current_time) 
  {
    spdlog::debug("Starting cache cleanup, current size: {}", query_cache.size());
    
    size_t removed_expired = 0;
    auto it = query_cache.begin();
    while (it != query_cache.end()) 
    {
      auto age = current_time - it->second.timestamp;
      if (age > CACHE_DURATION) 
      {
        it = query_cache.erase(it);
        removed_expired++;
      }
      else 
      {
        ++it;
      }
    }
    
    if (query_cache.size() > MAX_CACHE_SIZE) 
    {
      std::vector<std::pair<size_t, std::chrono::steady_clock::time_point>> entries;
      for (const auto& pair : query_cache) 
      {
        entries.push_back({pair.first, pair.second.timestamp});
      }
      
      std::sort(entries.begin(), entries.end(), 
                [](const auto& a, const auto& b) { return a.second < b.second; });
      
      size_t remove_count = query_cache.size() / 2;
      for (size_t i = 0; i < remove_count; ++i) 
      {
        query_cache.erase(entries[i].first);
      }
      
      spdlog::debug("Cache cleanup: removed {} expired, {} for size limit, final size: {}", 
                    removed_expired, remove_count, query_cache.size());
    }
    else 
    {
      spdlog::debug("Cache cleanup: removed {} expired entries, final size: {}", 
                    removed_expired, query_cache.size());
    }
  }
};

CollisionQuery::CollisionQuery(const GeometryConfig& config)
  : impl_(std::make_unique<Implementation>(config)) 
{
  math::simd::logSIMDInfo();
  spdlog::set_level(spdlog::level::debug);
}

CollisionQuery::~CollisionQuery() = default;

void CollisionQuery::setTerrain(std::shared_ptr<terrain::WaveletGrid> terrain) 
{
  impl_->terrain = terrain;
  spdlog::info("CollisionQuery: Terrain component set");
}

void CollisionQuery::setStaticObstacles(std::shared_ptr<static_obstacles::BoundingSphereTree> static_obs) 
{
  impl_->static_obstacles = static_obs;
  impl_->initializeStaticGrid();
  spdlog::info("CollisionQuery: Static obstacles component set");
}

void CollisionQuery::setDynamicObstacles(std::shared_ptr<dynamic_obstacles::SweptOBBBVH> dynamic_obs) 
{
  impl_->dynamic_obstacles = dynamic_obs;
  spdlog::info("CollisionQuery: Dynamic obstacles component set");
}

void CollisionQuery::setCorridors(std::shared_ptr<corridors::SpatialCorridorMap> corridors) 
{
  impl_->corridors = corridors;
  spdlog::info("CollisionQuery: Corridors component set");
}

HitResult CollisionQuery::queryCapsule(const Capsule& capsule,
                                      const std::chrono::steady_clock::time_point& time,
                                      UAVId uav_id) const 
{
  auto start_time = std::chrono::steady_clock::now();
  impl_->last_query_start = start_time;
  
  HitResult result;
  
  if (impl_->cache_enabled && uav_id == static_cast<UAVId>(INVALID_ID)) 
  {
    size_t key = impl_->computeCacheKey(capsule);
    
    std::lock_guard<std::mutex> cache_lock(impl_->cache_mutex);
    auto it = impl_->query_cache.find(key);
    if (it != impl_->query_cache.end()) 
    {
      auto age = start_time - it->second.timestamp;
      if (age < Implementation::CACHE_DURATION) 
      {
        result = it->second.result;
        spdlog::debug("Cache hit for capsule query, age: {}ms", 
                      std::chrono::duration_cast<std::chrono::milliseconds>(age).count());
        
        std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
        impl_->stats.last_query_time = 
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
      }
    }
    
    if (start_time - impl_->last_cache_cleanup > std::chrono::seconds(10)) 
    {
      impl_->cleanupCache(start_time);
      impl_->last_cache_cleanup = start_time;
    }
  }
  
  Eigen::Vector3f capsule_center = (capsule.p0 + capsule.p1) * 0.5f;
  auto terrain_result = impl_->checkTerrain(capsule_center, capsule.radius);
  if (terrain_result.hit) 
  {
    result = terrain_result;
    spdlog::debug("Terrain collision detected at height {:.2f}", terrain_result.hit_point.z());
    
    if (impl_->early_exit_enabled) 
    {
      if (impl_->cache_enabled && uav_id == static_cast<UAVId>(INVALID_ID)) 
      {
        std::lock_guard<std::mutex> cache_lock(impl_->cache_mutex);
        size_t key = impl_->computeCacheKey(capsule);
        impl_->query_cache[key] = {result, start_time};
      }
      
      std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
      impl_->stats.last_query_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_time);
      return result;
    }
  }
  
  if (uav_id != static_cast<UAVId>(INVALID_ID)) 
  {
    auto corridor_result = impl_->checkCorridors(capsule, uav_id);
    if (corridor_result.hit) 
    {
      spdlog::debug("Corridor violation detected for UAV {}", uav_id);
      
      if (!result.hit || corridor_result.t_min < result.t_min) 
      {
        result = corridor_result;
      }
      
      if (impl_->early_exit_enabled) 
      {
        if (impl_->cache_enabled && uav_id == static_cast<UAVId>(INVALID_ID)) 
        {
          std::lock_guard<std::mutex> cache_lock(impl_->cache_mutex);
          size_t key = impl_->computeCacheKey(capsule);
          impl_->query_cache[key] = {result, start_time};
        }
        
        std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
        impl_->stats.last_query_time = 
          std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time);
        return result;
      }
    }
  }
  
  auto static_result = impl_->checkStaticObstacles(capsule);
  if (static_result.hit) 
  {
    spdlog::debug("Static obstacle collision detected, object_id: {}", static_result.object_id);
    
    if (!result.hit || static_result.t_min < result.t_min) 
    {
      result = static_result;
    }
    
    if (impl_->early_exit_enabled) 
    {
      if (impl_->cache_enabled && uav_id == static_cast<UAVId>(INVALID_ID)) 
      {
        std::lock_guard<std::mutex> cache_lock(impl_->cache_mutex);
        size_t key = impl_->computeCacheKey(capsule);
        impl_->query_cache[key] = {result, start_time};
      }
      
      std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
      impl_->stats.last_query_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - start_time);
      return result;
    }
  }
  
  auto dynamic_result = impl_->checkDynamicObstacles(capsule, time, time);
  if (dynamic_result.hit) 
  {
    spdlog::debug("Dynamic obstacle collision detected at time {}", 
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                    time.time_since_epoch()).count());
    
    if (!result.hit || dynamic_result.t_min < result.t_min) 
    {
      result = dynamic_result;
    }
  }
  
  if (impl_->cache_enabled && uav_id == static_cast<UAVId>(INVALID_ID)) 
  {
    std::lock_guard<std::mutex> cache_lock(impl_->cache_mutex);
    size_t key = impl_->computeCacheKey(capsule);
    impl_->query_cache[key] = {result, start_time};
    spdlog::debug("Cached query result, hit: {}, type: {}", 
                  result.hit, static_cast<int>(result.object_type));
  }
  
  std::lock_guard<std::mutex> stats_lock(impl_->stats_mutex);
  impl_->stats.last_query_time = 
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  
  spdlog::debug("Query completed in {}us, hit: {}", 
                impl_->stats.last_query_time.count(), result.hit);
  
  return result;
}

HitResult CollisionQuery::queryCapsule(const Capsule& capsule,
                                      const std::chrono::steady_clock::time_point& t0,
                                      const std::chrono::steady_clock::time_point& t1,
                                      UAVId uav_id) const 
{
  auto start_time = std::chrono::steady_clock::now();
  
  HitResult result;
  
  result = queryCapsule(capsule, t0, uav_id);
  
  if (!result.hit || !impl_->early_exit_enabled) 
  {
    auto dynamic_result = impl_->checkDynamicObstacles(capsule, t0, t1);
    if (dynamic_result.hit && (!result.hit || dynamic_result.t_min < result.t_min)) 
    {
      result = dynamic_result;
    }
  }
  
  std::lock_guard<std::mutex> lock(impl_->stats_mutex);
  impl_->stats.last_query_time = 
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - start_time);
  
  return result;
}

HitResult CollisionQuery::querySegment(const LineSegment& segment,
                                      const std::chrono::steady_clock::time_point& time,
                                      UAVId uav_id) const 
{
  Capsule capsule;
  capsule.p0 = segment.start;
  capsule.p1 = segment.end;
  capsule.radius = 0.0f;
  
  return queryCapsule(capsule, time, uav_id);
}

HitResult CollisionQuery::querySegment(const LineSegment& segment,
                                      const std::chrono::steady_clock::time_point& t0,
                                      const std::chrono::steady_clock::time_point& t1,
                                      UAVId uav_id) const 
{
  Capsule capsule;
  capsule.p0 = segment.start;
  capsule.p1 = segment.end;
  capsule.radius = 0.0f;
  
  return queryCapsule(capsule, t0, t1, uav_id);
}

HitResult CollisionQuery::queryPath(const std::vector<Eigen::Vector3f>& path,
                                   float radius,
                                   const std::chrono::steady_clock::time_point& start_time,
                                   float velocity,
                                   UAVId uav_id) const 
{
  HitResult result;
  
  if (path.size() < 2 || velocity <= 0) 
  {
    return result;
  }
  
  auto current_time = start_time;
  
  for (size_t i = 0; i < path.size() - 1; ++i) 
  {
    Capsule segment_capsule;
    segment_capsule.p0 = path[i];
    segment_capsule.p1 = path[i + 1];
    segment_capsule.radius = radius;
    
    float segment_length = (path[i + 1] - path[i]).norm();
    auto segment_duration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<float>(segment_length / velocity));
    auto segment_end_time = current_time + segment_duration;
    
    auto segment_result = queryCapsule(segment_capsule, current_time, segment_end_time, uav_id);
    
    if (segment_result.hit) 
    {
      result = segment_result;
      float path_progress = 0.0f;
      for (size_t j = 0; j < i; ++j) 
      {
        path_progress += (path[j + 1] - path[j]).norm();
      }
      path_progress += segment_result.t_min * segment_length;
      
      float total_length = 0.0f;
      for (size_t j = 0; j < path.size() - 1; ++j) 
      {
        total_length += (path[j + 1] - path[j]).norm();
      }
      
      result.t_min = path_progress / total_length;
      break;
    }
    
    current_time = segment_end_time;
  }
  
  return result;
}

bool CollisionQuery::isFree(const Eigen::Vector3f& position,
                           float radius,
                           const std::chrono::steady_clock::time_point& time,
                           UAVId uav_id) const 
{
  Sphere sphere;
  sphere.center = position;
  sphere.radius = radius;
  
  Capsule capsule;
  capsule.p0 = position;
  capsule.p1 = position;
  capsule.radius = radius;
  
  auto result = queryCapsule(capsule, time, uav_id);
  return !result.hit;
}

float CollisionQuery::getDistance(const Eigen::Vector3f& position,
                                 const std::chrono::steady_clock::time_point& time,
                                 UAVId uav_id) const 
{
  float min_distance = std::numeric_limits<float>::max();
  
  if (impl_->terrain) 
  {
    float terrain_height = impl_->terrain->getHeight(position.x(), position.y());
    float terrain_distance = position.z() - terrain_height;
    min_distance = std::min(min_distance, terrain_distance);
  }
  
  if (impl_->static_obstacles) 
  {
    if (impl_->static_obstacles->isPointInside(position))
    {
      min_distance = std::min(min_distance, 0.0f);
    }
    else
    {
      float static_distance = impl_->static_obstacles->getDistance(position);
      min_distance = std::min(min_distance, static_distance);
    }
  }
  
  if (impl_->dynamic_obstacles) 
  {
    float dynamic_distance = impl_->dynamic_obstacles->getDistance(position, time);
    if (dynamic_distance < 0)
    {
      min_distance = std::min(min_distance, 0.0f);
    }
    else
    {
      min_distance = std::min(min_distance, dynamic_distance);
    }
  }
  
  if (impl_->corridors && uav_id != static_cast<UAVId>(INVALID_ID)) 
  {
    float corridor_distance = impl_->corridors->getDistanceToBoundary(position, uav_id);
    min_distance = std::min(min_distance, corridor_distance);
  }
  
  return min_distance;
}

void CollisionQuery::batchQueryCapsule(const Capsule* capsules,
                                      size_t count,
                                      const std::chrono::steady_clock::time_point& time,
                                      HitResult* results,
                                      UAVId uav_id) const 
{
  namespace simd = math::simd;
  
  if (count > 1000) 
  {
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, count),
      [&](const tbb::blocked_range<size_t>& range) 
      {
        for (size_t i = range.begin(); i != range.end(); ++i) 
        {
          results[i] = queryCapsule(capsules[i], time, uav_id);
        }
      }
    );
  }
  else 
  {
    simd::simdLoop(count, [&](size_t i, bool) 
    {
      results[i] = queryCapsule(capsules[i], time, uav_id);
    });
  }
}

std::vector<CollisionQuery::ObstacleInfo> CollisionQuery::getObstaclesInRadius(
    const Eigen::Vector3f& position,
    float radius,
    const std::chrono::steady_clock::time_point& time) const 
{
  std::vector<ObstacleInfo> obstacles;
  
  BoundingBox query_box;
  query_box.min = position - Eigen::Vector3f::Constant(radius);
  query_box.max = position + Eigen::Vector3f::Constant(radius);
  
  if (impl_->static_grid) 
  {
    auto candidates = impl_->static_grid->query(query_box);
    for (ObjectId id : candidates) 
    {
      ObstacleInfo info;
      info.id = id;
      info.type = HitResult::ObjectType::STATIC_OBSTACLE;
      info.distance = impl_->static_obstacles->getDistance(position);
      info.closest_point = position;
      info.bbox = query_box;
      obstacles.push_back(info);
    }
  }
  
  if (impl_->dynamic_obstacles) 
  {
    auto active_obbs = impl_->dynamic_obstacles->getActiveOBBs(time);
    for (const auto& obb : active_obbs) 
    {
      float dist = pointToOBBDistance(position, obb);
      if (dist <= radius) 
      {
        ObstacleInfo info;
        info.id = 0;
        info.type = HitResult::ObjectType::DYNAMIC_OBSTACLE;
        info.distance = dist;
        
        Eigen::Vector3f local_point = obb.orientation.transpose() * (position - obb.center);
        Eigen::Vector3f closest_local;
        for (int i = 0; i < 3; ++i) 
        {
          closest_local[i] = std::clamp(local_point[i], -obb.half_extents[i], obb.half_extents[i]);
        }
        info.closest_point = obb.center + obb.orientation * closest_local;
        
        Eigen::Matrix3f abs_orientation = obb.orientation.cwiseAbs();
        Eigen::Vector3f world_half_extents = abs_orientation * obb.half_extents;
        info.bbox.min = obb.center - world_half_extents;
        info.bbox.max = obb.center + world_half_extents;
        
        obstacles.push_back(info);
      }
    }
  }
  
  return obstacles;
}

CollisionQuery::PerformanceStats CollisionQuery::getPerformanceStats() const 
{
  std::lock_guard<std::mutex> lock(impl_->stats_mutex);
  return impl_->stats;
}

void CollisionQuery::resetPerformanceStats() 
{
  std::lock_guard<std::mutex> lock(impl_->stats_mutex);
  impl_->stats = PerformanceStats();
}

void CollisionQuery::setEarlyExitEnabled(bool enabled) 
{
  impl_->early_exit_enabled = enabled;
}

void CollisionQuery::setCacheEnabled(bool enabled) 
{
  impl_->cache_enabled = enabled;
  if (!enabled) 
  {
    std::lock_guard<std::mutex> lock(impl_->cache_mutex);
    impl_->query_cache.clear();
  }
}

void CollisionQuery::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["components"] = {
    {"terrain", impl_->terrain != nullptr},
    {"static_obstacles", impl_->static_obstacles != nullptr},
    {"dynamic_obstacles", impl_->dynamic_obstacles != nullptr},
    {"corridors", impl_->corridors != nullptr}
  };
  
  j["config"] = {
    {"grid_cell_size", impl_->config.grid_cell_size},
    {"hash_table_size", impl_->config.hash_table_size},
    {"early_exit_enabled", impl_->early_exit_enabled},
    {"cache_enabled", impl_->cache_enabled}
  };
  
  auto stats = getPerformanceStats();
  j["performance"] = {
    {"broad_phase_checks", stats.broad_phase_checks},
    {"narrow_phase_checks", stats.narrow_phase_checks},
    {"terrain_checks", stats.terrain_checks},
    {"static_checks", stats.static_checks},
    {"dynamic_checks", stats.dynamic_checks},
    {"corridor_checks", stats.corridor_checks},
    {"last_query_time_us", stats.last_query_time.count()}
  };
  
  {
    std::lock_guard<std::mutex> lock(impl_->cache_mutex);
    j["cache"] = {
      {"entries", impl_->query_cache.size()},
      {"enabled", impl_->cache_enabled},
      {"max_size", Implementation::MAX_CACHE_SIZE},
      {"duration_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
                         Implementation::CACHE_DURATION).count()}
    };
  }
  
  if (impl_->static_obstacles) 
  {
    auto bst_stats = impl_->static_obstacles->getStats();
    j["static_obstacles_stats"] = {
      {"total_nodes", bst_stats.total_nodes},
      {"leaf_nodes", bst_stats.leaf_nodes},
      {"max_depth", bst_stats.max_depth},
      {"avg_faces_per_leaf", bst_stats.average_faces_per_leaf}
    };
  }
  
  if (impl_->dynamic_obstacles) 
  {
    auto bvh_stats = impl_->dynamic_obstacles->getStats();
    j["dynamic_obstacles_stats"] = {
      {"num_objects", bvh_stats.num_objects},
      {"num_nodes", bvh_stats.num_nodes},
      {"tree_depth", bvh_stats.tree_depth},
      {"avg_obbs_per_leaf", bvh_stats.avg_obbs_per_leaf}
    };
  }
  
  if (impl_->corridors) 
  {
    auto corridor_stats = impl_->corridors->getStats();
    j["corridors_stats"] = {
      {"total_corridors", corridor_stats.total_corridors},
      {"total_segments", corridor_stats.total_segments},
      {"active_segments", corridor_stats.active_segments},
      {"freed_segments", corridor_stats.freed_segments}
    };
  }
  
  if (impl_->static_grid) 
  {
    auto grid_stats = impl_->static_grid->getStats();
    j["static_grid_stats"] = {
      {"total_objects", grid_stats.total_objects},
      {"occupied_cells", grid_stats.occupied_cells},
      {"avg_objects_per_cell", grid_stats.average_objects_per_cell},
      {"max_objects_in_cell", grid_stats.max_objects_in_cell}
    };
  }
  
  std::ofstream file(filename);
  file << j.dump(2);
  spdlog::info("Exported collision query state to {}", filename);
}

}
}
}