#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/geometry/intersections.hpp"
#include "vkr/geometry/morton.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace vkr
{
namespace geometry
{
namespace corridors
{

struct SpatialCorridorMap::Implementation 
{
  GeometryConfig config;
  tbb::concurrent_hash_map<CorridorId, Corridor> corridors;
  tbb::concurrent_hash_map<SegmentId, CorridorSegment> segments;
  std::vector<ConcurrentBucket> spatial_hash;
  
  std::atomic<CorridorId> next_corridor_id{1};
  std::atomic<SegmentId> next_segment_id{1};
  
  std::atomic<size_t> total_segments{0};
  std::atomic<size_t> active_segments{0};
  std::atomic<size_t> freed_segments{0};
  
  Implementation(const GeometryConfig& cfg) 
    : config(cfg), spatial_hash(cfg.hash_table_size) 
  {
    spdlog::debug("SpatialCorridorMap: cell_size={}, hash_table_size={}, coverage_fraction={}", 
                  config.grid_cell_size, config.hash_table_size, config.corridor_coverage_fraction);
  }
  
  uint32_t computeSpatialHash(const Eigen::Vector3f& position) const 
  {
    int32_t ix = static_cast<int32_t>(std::floor(position.x() / config.grid_cell_size));
    int32_t iy = static_cast<int32_t>(std::floor(position.y() / config.grid_cell_size));
    int32_t iz = static_cast<int32_t>(std::floor(position.z() / config.grid_cell_size));
    uint64_t morton = morton3d::encode(
      static_cast<uint32_t>(ix + 1048576),
      static_cast<uint32_t>(iy + 1048576),
      static_cast<uint32_t>(iz + 1048576)
    );
    morton ^= (morton >> 32);
    morton *= 0x9e3779b97f4a7c15ULL;
    morton ^= (morton >> 32);
    return static_cast<uint32_t>(morton % config.hash_table_size);
  }
  
  Eigen::Vector3i pointToGrid(const Eigen::Vector3f& point) const 
  {
    return Eigen::Vector3i(
      static_cast<int32_t>(std::floor(point.x() / config.grid_cell_size)),
      static_cast<int32_t>(std::floor(point.y() / config.grid_cell_size)),
      static_cast<int32_t>(std::floor(point.z() / config.grid_cell_size))
    );
  }
  
  std::vector<uint32_t> getCapsuleCells(const Capsule& capsule) const 
  {
    std::vector<uint32_t> cells;
    BoundingBox bbox;
    bbox.min = capsule.p0.cwiseMin(capsule.p1) - Eigen::Vector3f::Constant(capsule.radius);
    bbox.max = capsule.p0.cwiseMax(capsule.p1) + Eigen::Vector3f::Constant(capsule.radius);
    Eigen::Vector3i min_grid = pointToGrid(bbox.min);
    Eigen::Vector3i max_grid = pointToGrid(bbox.max);
    int estimated_cells = (max_grid.x() - min_grid.x() + 1) * 
                         (max_grid.y() - min_grid.y() + 1) * 
                         (max_grid.z() - min_grid.z() + 1);
    cells.reserve(std::min(estimated_cells, 256));
    
    for (int ix = min_grid.x(); ix <= max_grid.x(); ++ix) 
    {
      for (int iy = min_grid.y(); iy <= max_grid.y(); ++iy) 
      {
        for (int iz = min_grid.z(); iz <= max_grid.z(); ++iz) 
        {
          Eigen::Vector3f cell_min(ix * config.grid_cell_size, 
                                  iy * config.grid_cell_size, 
                                  iz * config.grid_cell_size);
          Eigen::Vector3f cell_max = cell_min + Eigen::Vector3f::Constant(config.grid_cell_size);
          BoundingBox cell_bbox{cell_min, cell_max};
          if (cell_bbox.intersects(bbox)) 
          {
            uint32_t hash = computeSpatialHash((cell_min + cell_max) * 0.5f);
            if (std::find(cells.begin(), cells.end(), hash) == cells.end()) 
            {
              cells.push_back(hash);
            }
          }
        }
      }
    }
    return cells;
  }
  
  void insertIntoSpatialHash(CorridorSegment& segment, const std::vector<uint32_t>& cells) 
  {
    for (uint32_t cell_hash : cells) 
    {
      auto& bucket = spatial_hash[cell_hash];
      uint32_t old_head = bucket.head.load(std::memory_order_relaxed);
      do 
      {
        segment.next.store(old_head, std::memory_order_relaxed);
      } while (!bucket.head.compare_exchange_weak(old_head, segment.segment_id, 
                                                  std::memory_order_release, 
                                                  std::memory_order_relaxed));
      bucket.size.fetch_add(1);
    }
  }
  
  void removeFromSpatialHash(SegmentId seg_id_to_remove, SegmentId next_id, 
                            const std::vector<uint32_t>& cells) 
  {
    for (uint32_t cell_hash : cells) 
    {
      auto& bucket = spatial_hash[cell_hash];
      uint32_t current_head = bucket.head.load();
      if (current_head == seg_id_to_remove) 
      {
        if (bucket.head.compare_exchange_strong(current_head, next_id)) 
        {
          if (bucket.size.load() > 0) bucket.size.fetch_sub(1);
        }
      }
    }
  }
  
  std::vector<uint32_t> getNeighboringCells(const Eigen::Vector3f& position, float radius) const 
  {
    std::vector<uint32_t> cells;
    int cell_radius = static_cast<int>(std::ceil(radius / config.grid_cell_size));
    Eigen::Vector3i center_grid = pointToGrid(position);
    
    for (int dx = -cell_radius; dx <= cell_radius; ++dx) 
    {
      for (int dy = -cell_radius; dy <= cell_radius; ++dy) 
      {
        for (int dz = -cell_radius; dz <= cell_radius; ++dz) 
        {
          Eigen::Vector3i grid_pos = center_grid + Eigen::Vector3i(dx, dy, dz);
          Eigen::Vector3f cell_center((grid_pos.x() + 0.5f) * config.grid_cell_size, 
                                     (grid_pos.y() + 0.5f) * config.grid_cell_size, 
                                     (grid_pos.z() + 0.5f) * config.grid_cell_size);
          uint32_t hash = computeSpatialHash(cell_center);
          if (std::find(cells.begin(), cells.end(), hash) == cells.end()) 
          {
            cells.push_back(hash);
          }
        }
      }
    }
    return cells;
  }
};

SpatialCorridorMap::SpatialCorridorMap(const GeometryConfig& config) 
  : impl_(std::make_unique<Implementation>(config)) 
{
}

SpatialCorridorMap::~SpatialCorridorMap() = default;

void SpatialCorridorMap::insertSegment(const CorridorSegment& segment) 
{
  typename decltype(impl_->segments)::accessor seg_acc;
  impl_->segments.insert(seg_acc, segment.segment_id);
  seg_acc->second.capsule = segment.capsule;
  seg_acc->second.corridor_id = segment.corridor_id;
  seg_acc->second.segment_id = segment.segment_id;
  seg_acc->second.next.store(segment.next.load());
  
  auto cells = impl_->getCapsuleCells(segment.capsule);
  impl_->insertIntoSpatialHash(seg_acc->second, cells);
  
  impl_->total_segments.fetch_add(1);
  impl_->active_segments.fetch_add(1);
}

void SpatialCorridorMap::removeSegment(SegmentId segment_id) 
{
  std::vector<uint32_t> cells;
  SegmentId next_id = INVALID_ID;
  {
    typename decltype(impl_->segments)::accessor segment_acc;
    if (impl_->segments.find(segment_acc, segment_id)) 
    {
      cells = impl_->getCapsuleCells(segment_acc->second.capsule);
      next_id = segment_acc->second.next.load();
      impl_->segments.erase(segment_acc);
    } 
    else 
    {
      return;
    }
  }

  if (!cells.empty()) 
  {
    impl_->removeFromSpatialHash(segment_id, next_id, cells);
  }
  
  impl_->active_segments.fetch_sub(1);
  impl_->freed_segments.fetch_add(1);
}

CorridorId SpatialCorridorMap::createCorridor(UAVId owner, const std::vector<Eigen::Vector3f>& path, 
                                              float radius, size_t current_waypoint_index) 
{
  if (path.size() < 2 || current_waypoint_index >= path.size() - 1) 
  {
    spdlog::warn("Cannot create corridor: invalid path or waypoint index");
    return INVALID_ID;
  }
  
  size_t total_remaining_segments = path.size() - 1 - current_waypoint_index;
  size_t segments_to_cover = std::max(size_t(1), 
                                     static_cast<size_t>(std::ceil(total_remaining_segments * 
                                                                  impl_->config.corridor_coverage_fraction)));
  segments_to_cover = std::min(segments_to_cover, total_remaining_segments);
  size_t end_waypoint = current_waypoint_index + segments_to_cover;
  
  CorridorId corridor_id = impl_->next_corridor_id.fetch_add(1);
  
  Corridor corridor;
  corridor.id = corridor_id;
  corridor.owner = owner;
  corridor.radius = radius;
  corridor.coverage_fraction = impl_->config.corridor_coverage_fraction;
  corridor.current_waypoint = current_waypoint_index;
  corridor.total_waypoints = path.size();
  
  for (size_t i = current_waypoint_index; i < end_waypoint; ++i) 
  {
    CorridorSegment segment;
    segment.segment_id = impl_->next_segment_id.fetch_add(1);
    segment.corridor_id = corridor_id;
    segment.capsule.p0 = path[i];
    segment.capsule.p1 = path[i + 1];
    segment.capsule.radius = radius;
    
    insertSegment(segment);
    corridor.active_segments.push_back(segment.segment_id);
  }
  
  typename decltype(impl_->corridors)::accessor corridor_acc;
  impl_->corridors.insert(corridor_acc, corridor_id);
  corridor_acc->second = std::move(corridor);
  
  spdlog::info("Created corridor {} for UAV {} covering {}/{} segments from waypoint {}", 
               corridor_id, owner, segments_to_cover, total_remaining_segments, 
               current_waypoint_index);
  
  return corridor_id;
}

void SpatialCorridorMap::updateCorridorCoverage(CorridorId corridor_id, 
                                               const std::vector<Eigen::Vector3f>& full_path, 
                                               size_t new_waypoint_index) 
{
  std::vector<SegmentId> segments_to_remove;
  std::vector<SegmentId> added_segment_ids;
  float corridor_radius = 0.0f;
  float coverage_fraction = 0.0f;
  bool corridor_should_be_removed = false;
  std::vector<SegmentId> kept_segments;

  {
    typename decltype(impl_->corridors)::const_accessor corridor_acc;
    if (!impl_->corridors.find(corridor_acc, corridor_id)) 
    {
      return;
    }
    
    const auto& corridor = corridor_acc->second;
    corridor_radius = corridor.radius;
    coverage_fraction = corridor.coverage_fraction;

    if (new_waypoint_index >= full_path.size() - 1) 
    {
      corridor_should_be_removed = true;
      segments_to_remove = corridor.active_segments;
    } 
    else 
    {
      for (SegmentId seg_id : corridor.active_segments) 
      {
        typename decltype(impl_->segments)::const_accessor segment_acc;
        bool is_behind = false;
        if (impl_->segments.find(segment_acc, seg_id)) 
        {
          for (size_t i = 0; i < new_waypoint_index && i + 1 < full_path.size(); ++i) 
          {
            if ((segment_acc->second.capsule.p1 - full_path[i+1]).squaredNorm() < GEOM_EPS) 
            {
              is_behind = true;
              break;
            }
          }
        }
        if (is_behind) segments_to_remove.push_back(seg_id);
        else kept_segments.push_back(seg_id);
      }
    }
  }

  for (SegmentId seg_id : segments_to_remove) 
  {
    removeSegment(seg_id);
  }

  if (corridor_should_be_removed) 
  {
    impl_->corridors.erase(corridor_id);
    spdlog::info("Removed corridor {} as UAV reached end of path", corridor_id);
    return;
  }

  size_t total_remaining = full_path.size() - 1 - new_waypoint_index;
  size_t should_cover = std::max(size_t(1), 
                                static_cast<size_t>(std::ceil(total_remaining * coverage_fraction)));
  
  if (kept_segments.size() < should_cover) 
  {
    size_t segments_to_add_start_idx = new_waypoint_index + kept_segments.size();
    size_t segments_to_add_end_idx = std::min(new_waypoint_index + should_cover, 
                                             full_path.size() - 1);
    
    for (size_t i = segments_to_add_start_idx; i < segments_to_add_end_idx; ++i) 
    {
      CorridorSegment segment;
      segment.segment_id = impl_->next_segment_id.fetch_add(1);
      segment.corridor_id = corridor_id;
      segment.capsule.p0 = full_path[i];
      segment.capsule.p1 = full_path[i + 1];
      segment.capsule.radius = corridor_radius;
      insertSegment(segment);
      added_segment_ids.push_back(segment.segment_id);
    }
  }

  {
    typename decltype(impl_->corridors)::accessor corridor_acc;
    if (impl_->corridors.find(corridor_acc, corridor_id)) 
    {
      corridor_acc->second.active_segments = kept_segments;
      corridor_acc->second.active_segments.insert(
        corridor_acc->second.active_segments.end(),
        added_segment_ids.begin(),
        added_segment_ids.end()
      );
      corridor_acc->second.current_waypoint = new_waypoint_index;
    }
  }
  
  spdlog::debug("Updated corridor {} at waypoint {}, now covering {} segments", 
                corridor_id, new_waypoint_index, kept_segments.size() + added_segment_ids.size());
}

void SpatialCorridorMap::freeSegment(CorridorId corridor_id, SegmentId segment_id) 
{
  {
    typename decltype(impl_->corridors)::accessor corridor_acc;
    if (impl_->corridors.find(corridor_acc, corridor_id)) 
    {
      auto& segs = corridor_acc->second.active_segments;
      segs.erase(std::remove(segs.begin(), segs.end(), segment_id), segs.end());
    }
  }
  removeSegment(segment_id);
  free_events_.push({corridor_id, segment_id, std::chrono::steady_clock::now()});
  spdlog::debug("Freed segment {} from corridor {}", segment_id, corridor_id);
}

void SpatialCorridorMap::removeCorridor(CorridorId corridor_id) 
{
  std::vector<SegmentId> segments_to_remove;
  {
    typename decltype(impl_->corridors)::accessor corridor_acc;
    if (impl_->corridors.find(corridor_acc, corridor_id)) 
    {
      segments_to_remove = corridor_acc->second.active_segments;
      impl_->corridors.erase(corridor_acc);
    }
  }
  for (SegmentId seg_id : segments_to_remove) 
  {
    removeSegment(seg_id);
  }
  spdlog::info("Removed corridor {} with {} segments", corridor_id, segments_to_remove.size());
}

bool SpatialCorridorMap::isInsideOwnCorridor(const Eigen::Vector3f& position, UAVId uav_id) const 
{
  auto cells = impl_->getNeighboringCells(position, impl_->config.corridor_radius);
  
  for (uint32_t hash : cells) 
  {
    const auto& bucket = impl_->spatial_hash[hash];
    uint32_t curr = bucket.head.load();
    while (curr != INVALID_ID) 
    {
      typename decltype(impl_->segments)::const_accessor segment_acc;
      if (impl_->segments.find(segment_acc, curr)) 
      {
        const auto& segment = segment_acc->second;
        typename decltype(impl_->corridors)::const_accessor corridor_acc;
        if (impl_->corridors.find(corridor_acc, segment.corridor_id)) 
        {
          if (corridor_acc->second.owner == uav_id) 
          {
            Eigen::Vector3f d = segment.capsule.p1 - segment.capsule.p0;
            float len_sq = d.squaredNorm();
            if (len_sq > GEOM_EPS) 
            {
              float t = std::clamp((position - segment.capsule.p0).dot(d) / len_sq, 0.0f, 1.0f);
              Eigen::Vector3f closest = segment.capsule.p0 + t * d;
              if ((position - closest).norm() <= segment.capsule.radius) 
              {
                return true;
              }
            } 
            else 
            {
              if ((position - segment.capsule.p0).norm() <= segment.capsule.radius) 
              {
                return true;
              }
            }
          }
        }
        curr = segment_acc->second.next.load();
      } 
      else 
      {
        break;
      }
    }
  }
  return false;
}

bool SpatialCorridorMap::isInsideOtherCorridor(const Eigen::Vector3f& position, UAVId uav_id) const 
{
  auto cells = impl_->getNeighboringCells(position, impl_->config.corridor_radius);
  
  for (uint32_t hash : cells) 
  {
    const auto& bucket = impl_->spatial_hash[hash];
    uint32_t curr = bucket.head.load();
    while (curr != INVALID_ID) 
    {
      typename decltype(impl_->segments)::const_accessor segment_acc;
      if (impl_->segments.find(segment_acc, curr)) 
      {
        const auto& segment = segment_acc->second;
        typename decltype(impl_->corridors)::const_accessor corridor_acc;
        if (impl_->corridors.find(corridor_acc, segment.corridor_id)) 
        {
          if (corridor_acc->second.owner != uav_id) 
          {
            Eigen::Vector3f d = segment.capsule.p1 - segment.capsule.p0;
            float len_sq = d.squaredNorm();
            if (len_sq > GEOM_EPS) 
            {
              float t = std::clamp((position - segment.capsule.p0).dot(d) / len_sq, 0.0f, 1.0f);
              Eigen::Vector3f closest = segment.capsule.p0 + t * d;
              if ((position - closest).norm() <= segment.capsule.radius) 
              {
                return true;
              }
            } 
            else 
            {
              if ((position - segment.capsule.p0).norm() <= segment.capsule.radius) 
              {
                return true;
              }
            }
          }
        }
        curr = segment_acc->second.next.load();
      } 
      else 
      {
        break;
      }
    }
  }
  return false;
}

float SpatialCorridorMap::getDistanceToBoundary(const Eigen::Vector3f& position, UAVId uav_id) const 
{
  float min_distance = std::numeric_limits<float>::max();
  float search_radius = impl_->config.corridor_radius * 3.0f;
  auto cells = impl_->getNeighboringCells(position, search_radius);
  
  for (uint32_t hash : cells) 
  {
    const auto& bucket = impl_->spatial_hash[hash];
    uint32_t curr = bucket.head.load();
    while (curr != INVALID_ID) 
    {
      typename decltype(impl_->segments)::const_accessor segment_acc;
      if (impl_->segments.find(segment_acc, curr)) 
      {
        const auto& segment = segment_acc->second;
        typename decltype(impl_->corridors)::const_accessor corridor_acc;
        if (impl_->corridors.find(corridor_acc, segment.corridor_id)) 
        {
          if (corridor_acc->second.owner == uav_id) 
          {
            Eigen::Vector3f d = segment.capsule.p1 - segment.capsule.p0;
            float len_sq = d.squaredNorm();
            if (len_sq > GEOM_EPS) 
            {
              float t = std::clamp((position - segment.capsule.p0).dot(d) / len_sq, 0.0f, 1.0f);
              float dist_to_center = (position - (segment.capsule.p0 + t * d)).norm();
              min_distance = std::min(min_distance, std::abs(segment.capsule.radius - dist_to_center));
            }
          }
        }
        curr = segment_acc->second.next.load();
      } 
      else 
      {
        break;
      }
    }
  }
  return min_distance;
}

bool SpatialCorridorMap::intersectCapsule(const Capsule& capsule, UAVId exclude_owner) const 
{
  auto cells = impl_->getCapsuleCells(capsule);
  for (uint32_t hash : cells) 
  {
    const auto& bucket = impl_->spatial_hash[hash];
    uint32_t curr = bucket.head.load();
    while (curr != INVALID_ID) 
    {
      typename decltype(impl_->segments)::const_accessor segment_acc;
      if (impl_->segments.find(segment_acc, curr)) 
      {
        const auto& segment = segment_acc->second;
        typename decltype(impl_->corridors)::const_accessor corridor_acc;
        if (impl_->corridors.find(corridor_acc, segment.corridor_id)) 
        {
          if (exclude_owner == INVALID_ID || corridor_acc->second.owner != exclude_owner) 
          {
            if (intersectCapsuleCapsule(capsule, segment.capsule)) 
            {
              return true;
            }
          }
        }
        curr = segment_acc->second.next.load();
      } 
      else 
      {
        break;
      }
    }
  }
  return false;
}

bool SpatialCorridorMap::intersectSegment(const LineSegment& segment, UAVId exclude_owner) const 
{
  Capsule capsule;
  capsule.p0 = segment.start;
  capsule.p1 = segment.end;
  capsule.radius = 0.0f;
  return intersectCapsule(capsule, exclude_owner);
}

std::vector<CorridorId> SpatialCorridorMap::getActiveCorridors(UAVId uav_id) const 
{
  std::vector<CorridorId> result;
  for (const auto& pair : impl_->corridors) 
  {
    if (pair.second.owner == uav_id && !pair.second.active_segments.empty()) 
    {
      result.push_back(pair.first);
    }
  }
  return result;
}

bool SpatialCorridorMap::getCorridorInfo(CorridorId corridor_id, Corridor& info) const 
{
  typename decltype(impl_->corridors)::const_accessor corridor_acc;
  if (impl_->corridors.find(corridor_acc, corridor_id)) 
  {
    info = corridor_acc->second;
    return true;
  }
  return false;
}

bool SpatialCorridorMap::getNextFreeEvent(CorridorFreeEvent& event) 
{
  return free_events_.try_pop(event);
}

void SpatialCorridorMap::batchCheckInside(const Eigen::Vector3f* positions, size_t count, 
                                          UAVId uav_id, bool* inside_own, bool* inside_other) const 
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
          inside_own[i] = isInsideOwnCorridor(positions[i], uav_id);
          inside_other[i] = isInsideOtherCorridor(positions[i], uav_id);
        }
      }
    );
  } 
  else 
  {
    simd::simdLoop(count, [&](size_t i, bool) 
    {
      inside_own[i] = isInsideOwnCorridor(positions[i], uav_id);
      inside_other[i] = isInsideOtherCorridor(positions[i], uav_id);
    });
  }
}

SpatialCorridorMap::Stats SpatialCorridorMap::getStats() const 
{
  Stats stats;
  stats.total_corridors = impl_->corridors.size();
  stats.total_segments = impl_->total_segments.load();
  stats.active_segments = impl_->active_segments.load();
  stats.freed_segments = impl_->freed_segments.load();
  return stats;
}

void SpatialCorridorMap::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  auto stats = getStats();
  j["stats"] = {
    {"total_corridors", stats.total_corridors},
    {"total_segments", stats.total_segments},
    {"active_segments", stats.active_segments},
    {"freed_segments", stats.freed_segments}
  };
  
  j["config"] = {
    {"grid_cell_size", impl_->config.grid_cell_size},
    {"hash_table_size", impl_->config.hash_table_size},
    {"corridor_coverage_fraction", impl_->config.corridor_coverage_fraction}
  };
  
  j["corridors"] = nlohmann::json::array();
  for (const auto& pair : impl_->corridors) 
  {
    const auto& corridor = pair.second;
    nlohmann::json corridor_json;
    corridor_json["id"] = corridor.id;
    corridor_json["owner"] = corridor.owner;
    corridor_json["radius"] = corridor.radius;
    corridor_json["coverage_fraction"] = corridor.coverage_fraction;
    corridor_json["current_waypoint"] = corridor.current_waypoint;
    corridor_json["total_waypoints"] = corridor.total_waypoints;
    corridor_json["num_active_segments"] = corridor.active_segments.size();
    corridor_json["remaining_path_fraction"] = corridor.getRemainingPathFraction();
    corridor_json["segments"] = nlohmann::json::array();
    
    for (SegmentId seg_id : corridor.active_segments) 
    {
      typename decltype(impl_->segments)::const_accessor segment_acc;
      if (impl_->segments.find(segment_acc, seg_id)) 
      {
        const auto& segment = segment_acc->second;
        nlohmann::json seg_json;
        seg_json["id"] = segment.segment_id;
        seg_json["p0"] = {segment.capsule.p0.x(), segment.capsule.p0.y(), segment.capsule.p0.z()};
        seg_json["p1"] = {segment.capsule.p1.x(), segment.capsule.p1.y(), segment.capsule.p1.z()};
        seg_json["radius"] = segment.capsule.radius;
        corridor_json["segments"].push_back(seg_json);
      }
    }
    j["corridors"].push_back(corridor_json);
  }
  
  j["spatial_hash"] = {
    {"table_size", impl_->config.hash_table_size},
    {"cell_size", impl_->config.grid_cell_size},
    {"non_empty_buckets", 0},
    {"max_bucket_size", 0},
    {"avg_bucket_size", 0.0}
  };
  
  size_t non_empty = 0;
  size_t max_size = 0;
  size_t total_size = 0;
  for (const auto& bucket : impl_->spatial_hash) 
  {
    size_t bucket_size = bucket.size.load();
    if (bucket_size > 0) 
    {
      non_empty++;
      max_size = std::max(max_size, bucket_size);
      total_size += bucket_size;
    }
  }
  j["spatial_hash"]["non_empty_buckets"] = non_empty;
  j["spatial_hash"]["max_bucket_size"] = max_size;
  if (non_empty > 0) 
  {
    j["spatial_hash"]["avg_bucket_size"] = static_cast<double>(total_size) / non_empty;
  }
  
  std::ofstream file(filename);
  file << j.dump(2);
  spdlog::info("Exported corridor map to {}", filename);
}

uint32_t SpatialCorridorMap::getHashKey(const Eigen::Vector3f& position, CorridorId, SegmentId) const 
{
  return impl_->computeSpatialHash(position);
}

}
}
}