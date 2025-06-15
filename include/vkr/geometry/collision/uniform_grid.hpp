#ifndef VKR_GEOMETRY_COLLISION_UNIFORM_GRID_HPP_
#define VKR_GEOMETRY_COLLISION_UNIFORM_GRID_HPP_

#include "vkr/geometry/primitives.hpp"
#include "vkr/common.hpp"
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <tbb/concurrent_vector.h>

namespace vkr
{
namespace geometry
{
namespace collision
{

struct GridCell
{
  std::atomic<uint32_t> head{INVALID_ID};
  std::atomic<uint32_t> count{0};
};

class UniformGrid
{
public:
  UniformGrid(float cell_size, const BoundingBox& bounds, uint32_t hash_table_size);
  ~UniformGrid();
  
  void insert(ObjectId id, const BoundingBox& bbox);
  
  void remove(ObjectId id);
  
  std::vector<ObjectId> query(const BoundingBox& bbox) const;
  
  std::vector<ObjectId> queryRay(const Eigen::Vector3f& origin,
                                 const Eigen::Vector3f& direction,
                                 float max_distance) const;
  
  ObjectId queryRayFirstHit(const Eigen::Vector3f& origin,
                           const Eigen::Vector3f& direction,
                           float max_distance) const;
  
  void update(ObjectId id, const BoundingBox& new_bbox);
  
  void clear();
  
  struct Stats
  {
    size_t total_objects;
    size_t occupied_cells;
    float average_objects_per_cell;
    size_t max_objects_in_cell;
  };
  
  Stats getStats() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct ObjectEntry
  {
    ObjectId id;
    BoundingBox bbox;
    std::vector<uint32_t> cell_indices;
    std::atomic<uint32_t> next{INVALID_ID};
  };
  
  float cell_size_;
  BoundingBox bounds_;
  uint32_t hash_mask_;
  tbb::concurrent_vector<GridCell> cells_;
  tbb::concurrent_vector<ObjectEntry> objects_;
  std::vector<std::atomic<uint32_t>> free_list_;
  std::unordered_map<ObjectId, size_t> object_index_;
  mutable std::mutex index_mutex_;
  
  std::atomic<size_t> next_object_index_{0};
  
  uint32_t getCellHash(int ix, int iy, int iz) const;
  void getCellIndices(const BoundingBox& bbox,
                     std::vector<uint32_t>& indices) const;
  Eigen::Vector3i worldToGrid(const Eigen::Vector3f& pos) const;
};

}
}
}

#endif