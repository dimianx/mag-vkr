#include "vkr/geometry/collision/uniform_grid.hpp"
#include "vkr/geometry/morton.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vkr
{
namespace geometry
{
namespace collision
{

UniformGrid::UniformGrid(float cell_size, const BoundingBox& bounds, uint32_t hash_table_size)
  : cell_size_(cell_size)
  , bounds_(bounds)
  , hash_mask_(hash_table_size - 1)
{
  if ((hash_table_size & (hash_table_size - 1)) != 0) 
  {
    spdlog::error("UniformGrid: hash_table_size must be power of 2, got {}", hash_table_size);
    hash_table_size = 1 << static_cast<uint32_t>(std::ceil(std::log2(hash_table_size)));
    hash_mask_ = hash_table_size - 1;
  }
  
  cells_.resize(hash_table_size);
  
  spdlog::info("UniformGrid: Created with cell_size={}, bounds=[{},{},{} to {},{},{}], hash_size={}", 
               cell_size_, bounds_.min.x(), bounds_.min.y(), bounds_.min.z(),
               bounds_.max.x(), bounds_.max.y(), bounds_.max.z(), hash_table_size);
}

UniformGrid::~UniformGrid() = default;

void UniformGrid::insert(ObjectId id, const BoundingBox& bbox)
{
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    if (object_index_.find(id) != object_index_.end()) 
    {
      spdlog::warn("UniformGrid: Object {} already exists, skipping insert", id);
      return;
    }
  }

  size_t obj_idx = next_object_index_.fetch_add(1, std::memory_order_relaxed);
  
  objects_.grow_to_at_least(obj_idx + 1);
  
  ObjectEntry& entry = objects_[obj_idx];
  
  entry.id = id;
  entry.bbox = bbox;
  entry.next.store(INVALID_ID, std::memory_order_relaxed);
  
  getCellIndices(bbox, entry.cell_indices);
  
  for (uint32_t cell_idx : entry.cell_indices) 
  {
    GridCell& cell = cells_[cell_idx];
    
    uint32_t old_head = cell.head.load(std::memory_order_relaxed);
    do 
    {
      entry.next.store(old_head, std::memory_order_relaxed);
    } 
    while (!cell.head.compare_exchange_weak(old_head, static_cast<uint32_t>(obj_idx),
                                           std::memory_order_release,
                                           std::memory_order_relaxed));
    
    cell.count.fetch_add(1, std::memory_order_relaxed);
  }
  
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    object_index_[id] = obj_idx;
  }
}

void UniformGrid::remove(ObjectId id)
{
  size_t obj_idx;
  
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    auto it = object_index_.find(id);
    if (it == object_index_.end()) 
    {
      return;
    }
    obj_idx = it->second;
    object_index_.erase(it);
  }
  
  ObjectEntry& entry = objects_[obj_idx];
  
  for (uint32_t cell_idx : entry.cell_indices) 
  {
    GridCell& cell = cells_[cell_idx];
    
    uint32_t prev = INVALID_ID;
    uint32_t curr = cell.head.load(std::memory_order_acquire);
    
    while (curr != INVALID_ID) 
    {
      if (curr == obj_idx) 
      {
        if (prev == INVALID_ID) 
        {
          cell.head.store(entry.next.load(std::memory_order_relaxed), 
                         std::memory_order_release);
        } 
        else 
        {
          uint32_t next_val = entry.next.load(std::memory_order_relaxed);
          objects_[prev].next.store(next_val, std::memory_order_release);
        }
        
        cell.count.fetch_sub(1, std::memory_order_relaxed);
        break;
      }
      
      prev = curr;
      curr = objects_[curr].next.load(std::memory_order_acquire);
    }
  }
  
  entry.id = INVALID_ID;
  entry.cell_indices.clear();
}

std::vector<ObjectId> UniformGrid::query(const BoundingBox& bbox) const
{
  std::vector<ObjectId> results;
  std::vector<uint32_t> cell_indices;
  getCellIndices(bbox, cell_indices);
  
  std::unordered_set<ObjectId> unique_results;
  
  for (uint32_t cell_idx : cell_indices) 
  {
    const GridCell& cell = cells_[cell_idx];
    uint32_t curr = cell.head.load(std::memory_order_acquire);
    
    while (curr != INVALID_ID && curr < objects_.size()) 
    {
      const ObjectEntry& entry = objects_[curr];
      
      if (entry.id != INVALID_ID && entry.bbox.intersects(bbox)) 
      {
        unique_results.insert(entry.id);
      }
      
      curr = entry.next.load(std::memory_order_acquire);
    }
  }
  
  results.assign(unique_results.begin(), unique_results.end());
  return results;
}

std::vector<ObjectId> UniformGrid::queryRay(const Eigen::Vector3f& origin,
                                           const Eigen::Vector3f& direction,
                                           float max_distance) const
{
  std::vector<ObjectId> results;
  std::unordered_set<ObjectId> unique_results;
  
  Eigen::Vector3i grid_start = worldToGrid(origin);
  
  Eigen::Vector3i step;
  Eigen::Vector3f t_delta;
  Eigen::Vector3f t_max;
  
  for (int i = 0; i < 3; ++i) 
  {
    if (std::abs(direction[i]) < GEOM_EPS) 
    {
      step[i] = 0;
      t_delta[i] = std::numeric_limits<float>::max();
      t_max[i] = std::numeric_limits<float>::max();
    } 
    else 
    {
      step[i] = direction[i] > 0 ? 1 : -1;
      t_delta[i] = cell_size_ / std::abs(direction[i]);
      
      float voxel_boundary = (grid_start[i] + (step[i] > 0 ? 1 : 0)) * cell_size_ + bounds_.min[i];
      t_max[i] = (voxel_boundary - origin[i]) / direction[i];
    }
  }
  
  Eigen::Vector3i current_grid = grid_start;
  float t = 0.0f;
  
  while (t < max_distance) 
  {
    uint32_t cell_hash = getCellHash(current_grid.x(), current_grid.y(), current_grid.z());
    const GridCell& cell = cells_[cell_hash];
    
    uint32_t curr = cell.head.load(std::memory_order_acquire);
    while (curr != INVALID_ID && curr < objects_.size()) 
    {
      const ObjectEntry& entry = objects_[curr];
      if (entry.id != INVALID_ID) 
      {
        float t_near = 0.0f;
        float t_far = max_distance;
        bool hit = true;
        
        for (int i = 0; i < 3; ++i) 
        {
          if (std::abs(direction[i]) < GEOM_EPS) 
          {
            if (origin[i] < entry.bbox.min[i] || origin[i] > entry.bbox.max[i]) 
            {
              hit = false;
              break;
            }
          } 
          else 
          {
            float inv_d = 1.0f / direction[i];
            float t1 = (entry.bbox.min[i] - origin[i]) * inv_d;
            float t2 = (entry.bbox.max[i] - origin[i]) * inv_d;
            
            if (t1 > t2) std::swap(t1, t2);
            
            t_near = std::max(t_near, t1);
            t_far = std::min(t_far, t2);
            
            if (t_near > t_far || t_far < 0.0f) 
            {
              hit = false;
              break;
            }
          }
        }
        
        if (hit && t_near <= max_distance) 
        {
          unique_results.insert(entry.id);
        }
      }
      curr = entry.next.load(std::memory_order_acquire);
    }
    
    if (t_max.x() < t_max.y() && t_max.x() < t_max.z()) 
    {
      current_grid.x() += step.x();
      t = t_max.x();
      t_max.x() += t_delta.x();
    } 
    else if (t_max.y() < t_max.z()) 
    {
      current_grid.y() += step.y();
      t = t_max.y();
      t_max.y() += t_delta.y();
    } 
    else 
    {
      current_grid.z() += step.z();
      t = t_max.z();
      t_max.z() += t_delta.z();
    }
    
    if (current_grid.x() < 0 || current_grid.y() < 0 || current_grid.z() < 0) 
    {
      break;
    }
  }
  
  results.assign(unique_results.begin(), unique_results.end());
  return results;
}

ObjectId UniformGrid::queryRayFirstHit(const Eigen::Vector3f& origin,
                                     const Eigen::Vector3f& direction,
                                     float max_distance) const
{
  ObjectId closest_id = INVALID_ID;
  float closest_t = max_distance;
  
  Eigen::Vector3i grid_start = worldToGrid(origin);
  
  Eigen::Vector3i step;
  Eigen::Vector3f t_delta;
  Eigen::Vector3f t_max;
  
  for (int i = 0; i < 3; ++i) 
  {
    if (std::abs(direction[i]) < GEOM_EPS) 
    {
      step[i] = 0;
      t_delta[i] = std::numeric_limits<float>::max();
      t_max[i] = std::numeric_limits<float>::max();
    } 
    else 
    {
      step[i] = direction[i] > 0 ? 1 : -1;
      t_delta[i] = cell_size_ / std::abs(direction[i]);
      
      float voxel_boundary = (grid_start[i] + (step[i] > 0 ? 1 : 0)) * cell_size_ + bounds_.min[i];
      t_max[i] = (voxel_boundary - origin[i]) / direction[i];
    }
  }
  
  Eigen::Vector3i current_grid = grid_start;
  float t = 0.0f;
  
  while (t < closest_t) 
  {
    uint32_t cell_hash = getCellHash(current_grid.x(), current_grid.y(), current_grid.z());
    const GridCell& cell = cells_[cell_hash];
    
    uint32_t curr = cell.head.load(std::memory_order_acquire);
    while (curr != INVALID_ID && curr < objects_.size()) 
    {
      const ObjectEntry& entry = objects_[curr];
      if (entry.id != INVALID_ID) 
      {
        float t_near = 0.0f;
        float t_far = closest_t;
        bool hit = true;
        
        for (int i = 0; i < 3; ++i) 
        {
          if (std::abs(direction[i]) < GEOM_EPS) 
          {
            if (origin[i] < entry.bbox.min[i] || origin[i] > entry.bbox.max[i]) 
            {
              hit = false;
              break;
            }
          } 
          else 
          {
            float inv_d = 1.0f / direction[i];
            float t1 = (entry.bbox.min[i] - origin[i]) * inv_d;
            float t2 = (entry.bbox.max[i] - origin[i]) * inv_d;
            
            if (t1 > t2) std::swap(t1, t2);
            
            t_near = std::max(t_near, t1);
            t_far = std::min(t_far, t2);
            
            if (t_near > t_far || t_far < 0.0f) 
            {
              hit = false;
              break;
            }
          }
        }
        
        if (hit && t_near < closest_t) 
        {
          closest_t = t_near;
          closest_id = entry.id;
        }
      }
      curr = entry.next.load(std::memory_order_acquire);
    }
    
    if (closest_id != INVALID_ID && t > closest_t) 
    {
      break;
    }
    
    if (t_max.x() < t_max.y() && t_max.x() < t_max.z()) 
    {
      current_grid.x() += step.x();
      t = t_max.x();
      t_max.x() += t_delta.x();
    } 
    else if (t_max.y() < t_max.z()) 
    {
      current_grid.y() += step.y();
      t = t_max.y();
      t_max.y() += t_delta.y();
    } 
    else 
    {
      current_grid.z() += step.z();
      t = t_max.z();
      t_max.z() += t_delta.z();
    }
    
    if (current_grid.x() < 0 || current_grid.y() < 0 || current_grid.z() < 0) 
    {
      break;
    }
  }
  
  return closest_id;
}

void UniformGrid::update(ObjectId id, const BoundingBox& new_bbox)
{
  remove(id);
  insert(id, new_bbox);
}

void UniformGrid::clear()
{
  for (size_t i = 0; i < cells_.size(); ++i) 
  {
    cells_[i].head.store(INVALID_ID, std::memory_order_relaxed);
    cells_[i].count.store(0, std::memory_order_relaxed);
  }
  
  objects_.clear();
  
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    object_index_.clear();
  }
  
  next_object_index_.store(0, std::memory_order_relaxed);
}

UniformGrid::Stats UniformGrid::getStats() const
{
  Stats stats{};
  
  {
    std::lock_guard<std::mutex> lock(index_mutex_);
    stats.total_objects = object_index_.size();
  }
  
  stats.occupied_cells = 0;
  stats.max_objects_in_cell = 0;
  
  size_t total_objects_in_cells = 0;
  
  for (size_t i = 0; i < cells_.size(); ++i) 
  {
    uint32_t count = cells_[i].count.load(std::memory_order_relaxed);
    if (count > 0) 
    {
      stats.occupied_cells++;
      total_objects_in_cells += count;
      stats.max_objects_in_cell = std::max(stats.max_objects_in_cell, static_cast<size_t>(count));
    }
  }
  
  if (stats.occupied_cells > 0) 
  {
    stats.average_objects_per_cell = static_cast<float>(total_objects_in_cells) / stats.occupied_cells;
  }
  
  return stats;
}

void UniformGrid::exportToJSON(const std::string& filename) const
{
  nlohmann::json j;
  
  j["config"] = 
  {
    {"cell_size", cell_size_},
    {"hash_table_size", cells_.size()},
    {"bounds", 
     {
       {"min", {bounds_.min.x(), bounds_.min.y(), bounds_.min.z()}},
       {"max", {bounds_.max.x(), bounds_.max.y(), bounds_.max.z()}}
     }}
  };
  
  Stats stats = getStats();
  j["stats"] = 
  {
    {"total_objects", stats.total_objects},
    {"occupied_cells", stats.occupied_cells},
    {"average_objects_per_cell", stats.average_objects_per_cell},
    {"max_objects_in_cell", stats.max_objects_in_cell}
  };
  
  nlohmann::json cells_json = nlohmann::json::array();
  size_t exported_cells = 0;
  size_t max_cells_to_export = 1000;
  
  for (size_t i = 0; i < cells_.size() && exported_cells < max_cells_to_export; ++i) 
  {
    const GridCell& cell = cells_[i];
    uint32_t count = cell.count.load(std::memory_order_relaxed);
    
    if (count > 0) 
    {
      nlohmann::json cell_json;
      cell_json["hash_index"] = i;
      cell_json["object_count"] = count;
      
      nlohmann::json objects_json = nlohmann::json::array();
      uint32_t curr = cell.head.load(std::memory_order_acquire);
      
      while (curr != INVALID_ID && curr < objects_.size()) 
      {
        const ObjectEntry& entry = objects_[curr];
        if (entry.id != INVALID_ID) 
        {
          nlohmann::json obj_json;
          obj_json["id"] = entry.id;
          obj_json["bbox"] = 
          {
            {"min", {entry.bbox.min.x(), entry.bbox.min.y(), entry.bbox.min.z()}},
            {"max", {entry.bbox.max.x(), entry.bbox.max.y(), entry.bbox.max.z()}}
          };
          objects_json.push_back(obj_json);
        }
        curr = entry.next.load(std::memory_order_acquire);
      }
      
      cell_json["objects"] = objects_json;
      cells_json.push_back(cell_json);
      exported_cells++;
    }
  }
  
  j["cells_sample"] = cells_json;
  j["cells_exported"] = exported_cells;
  
  std::ofstream file(filename);
  file << j.dump(2);
}

uint32_t UniformGrid::getCellHash(int ix, int iy, int iz) const
{
  uint32_t morton = morton3d::encode(
    static_cast<uint32_t>(ix & 0x3FF),
    static_cast<uint32_t>(iy & 0x3FF),
    static_cast<uint32_t>(iz & 0x3FF)
  );
  
  return morton & hash_mask_;
}

void UniformGrid::getCellIndices(const BoundingBox& bbox, 
                                std::vector<uint32_t>& indices) const
{
  indices.clear();
  
  std::unordered_set<uint32_t> unique_indices;
  
  Eigen::Vector3i min_grid = worldToGrid(bbox.min);
  Eigen::Vector3i max_grid = worldToGrid(bbox.max);
  
  for (int iz = min_grid.z(); iz <= max_grid.z(); ++iz)
  {
    for (int iy = min_grid.y(); iy <= max_grid.y(); ++iy)
    {
      for (int ix = min_grid.x(); ix <= max_grid.x(); ++ix)
      {
        uint32_t hash = getCellHash(ix, iy, iz);
        unique_indices.insert(hash);
      }
    }
  }
  
  indices.assign(unique_indices.begin(), unique_indices.end());
}

Eigen::Vector3i UniformGrid::worldToGrid(const Eigen::Vector3f& pos) const
{
  Eigen::Vector3f relative = pos - bounds_.min;
  Eigen::Vector3i grid_pos;
  
  for (int i = 0; i < 3; ++i) 
  {
    grid_pos[i] = static_cast<int>(std::floor(relative[i] / cell_size_));
  }
  
  return grid_pos;
}

}
}
}