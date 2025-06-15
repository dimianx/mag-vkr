#ifndef VKR_GEOMETRY_CORRIDORS_SPATIAL_CORRIDOR_MAP_HPP_
#define VKR_GEOMETRY_CORRIDORS_SPATIAL_CORRIDOR_MAP_HPP_

#include "vkr/geometry/corridors/corridor.hpp"
#include "vkr/geometry/corridors/corridor_segment.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <tbb/concurrent_queue.h>

namespace vkr
{
namespace geometry
{
namespace corridors
{

class SpatialCorridorMap
{
public:
  explicit SpatialCorridorMap(const GeometryConfig& config);
  ~SpatialCorridorMap();
  
  CorridorId createCorridor(UAVId owner,
                           const std::vector<Eigen::Vector3f>& path,
                           float radius,
                           size_t current_waypoint_index = 0);
  
  void updateCorridorCoverage(CorridorId corridor_id,
                             const std::vector<Eigen::Vector3f>& full_path,
                             size_t new_waypoint_index);
  
  void freeSegment(CorridorId corridor_id, SegmentId segment_id);
  
  void removeCorridor(CorridorId corridor_id);
  
  bool isInsideOwnCorridor(const Eigen::Vector3f& position,
                          UAVId uav_id) const;
  
  bool isInsideOtherCorridor(const Eigen::Vector3f& position,
                            UAVId uav_id) const;
  
  float getDistanceToBoundary(const Eigen::Vector3f& position,
                             UAVId uav_id) const;
  
  bool intersectCapsule(const Capsule& capsule,
                       UAVId exclude_owner = static_cast<UAVId>(INVALID_ID)) const;
  
  bool intersectSegment(const LineSegment& segment,
                       UAVId exclude_owner = static_cast<UAVId>(INVALID_ID)) const;
  
  std::vector<CorridorId> getActiveCorridors(UAVId uav_id) const;
  
  bool getCorridorInfo(CorridorId corridor_id, Corridor& info) const;
  
  bool getNextFreeEvent(CorridorFreeEvent& event);
  
  void batchCheckInside(const Eigen::Vector3f* positions,
                       size_t count,
                       UAVId uav_id,
                       bool* inside_own,
                       bool* inside_other) const;
  
  struct Stats
  {
    size_t total_corridors;
    size_t total_segments;
    size_t active_segments;
    size_t freed_segments;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct ConcurrentBucket
  {
    std::atomic<uint32_t> head{INVALID_ID};
    std::atomic<uint32_t> size{0};
  };
  
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  tbb::concurrent_queue<CorridorFreeEvent> free_events_;
  
  uint32_t getHashKey(const Eigen::Vector3f& position,
                     CorridorId cid, SegmentId seg_id) const;
  void insertSegment(const CorridorSegment& segment);
  void removeSegment(SegmentId segment_id);
};

}
}
}

#endif