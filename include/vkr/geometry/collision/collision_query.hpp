#ifndef VKR_GEOMETRY_COLLISION_COLLISION_QUERY_HPP_
#define VKR_GEOMETRY_COLLISION_COLLISION_QUERY_HPP_

#include "vkr/geometry/collision/hit_result.hpp"
#include "vkr/geometry/collision/uniform_grid.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <chrono>

namespace vkr
{

namespace terrain
{
class WaveletGrid;
}

namespace geometry
{

namespace static_obstacles
{
class BoundingSphereTree;
}

namespace corridors
{
class SpatialCorridorMap;
}

namespace collision
{

class CollisionQuery
{
public:
  explicit CollisionQuery(const GeometryConfig& config);
  ~CollisionQuery();
  
  void setTerrain(std::shared_ptr<terrain::WaveletGrid> terrain);
  void setStaticObstacles(std::shared_ptr<static_obstacles::BoundingSphereTree> static_obs);
  void setCorridors(std::shared_ptr<corridors::SpatialCorridorMap> corridors);
  
  HitResult queryCapsule(const Capsule& capsule,
                        const std::chrono::steady_clock::time_point& time,
                        UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  HitResult queryCapsule(const Capsule& capsule,
                        const std::chrono::steady_clock::time_point& t0,
                        const std::chrono::steady_clock::time_point& t1,
                        UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  HitResult querySegment(const LineSegment& segment,
                        const std::chrono::steady_clock::time_point& time,
                        UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  HitResult querySegment(const LineSegment& segment,
                        const std::chrono::steady_clock::time_point& t0,
                        const std::chrono::steady_clock::time_point& t1,
                        UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  HitResult queryPath(const std::vector<Eigen::Vector3f>& path,
                     float radius,
                     const std::chrono::steady_clock::time_point& start_time,
                     float velocity,
                     UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  bool isFree(const Eigen::Vector3f& position,
             float radius,
             const std::chrono::steady_clock::time_point& time,
             UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  float getDistance(const Eigen::Vector3f& position,
                   const std::chrono::steady_clock::time_point& time,
                   UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  void batchQueryCapsule(const Capsule* capsules,
                        size_t count,
                        const std::chrono::steady_clock::time_point& time,
                        HitResult* results,
                        UAVId uav_id = static_cast<UAVId>(INVALID_ID)) const;
  
  struct ObstacleInfo
  {
    ObjectId id;
    HitResult::ObjectType type;
    float distance;
    Eigen::Vector3f closest_point;
    BoundingBox bbox;
  };
  
  std::vector<ObstacleInfo> getObstaclesInRadius(
      const Eigen::Vector3f& position,
      float radius,
      const std::chrono::steady_clock::time_point& time) const;
  
  struct PerformanceStats
  {
    size_t broad_phase_checks = 0;
    size_t narrow_phase_checks = 0;
    size_t terrain_checks = 0;
    size_t static_checks = 0;
    size_t corridor_checks = 0;
    std::chrono::microseconds last_query_time{0};
  };
  
  PerformanceStats getPerformanceStats() const;
  void resetPerformanceStats();
  
  void setEarlyExitEnabled(bool enabled);
  
  void exportToJSON(const std::string& filename) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

}
}
}

#endif