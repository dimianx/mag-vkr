#ifndef VKR_CONFIG_HPP_
#define VKR_CONFIG_HPP_

#include <cstdint>
#include <cstddef>

namespace vkr
{

struct TerrainConfig
{
  uint8_t wavelet_levels = 8;
  size_t page_size = 4096;
  size_t max_cache_size = 1024;
};

struct GeometryConfig
{
  uint32_t max_faces_per_leaf = 32;
  
  uint32_t bvh_max_leaf_size = 4;
  bool bvh_use_temporal_splits = true;
  
  float corridor_radius = 2.5f;
  float corridor_coverage_fraction = 0.8f;
  
  float grid_cell_size = 8.0f;
  uint32_t hash_table_size = 65536;
};

struct PlanningConfig
{
  float coarse_grid_size = 4.0f;
  float fine_grid_size = 1.0f;
  float initial_epsilon = 2.5f;
  float corridor_penalty_weight = 1000.0f;
  float height_penalty_weight = 10.0f;
  float max_runtime_ms = 100.0f;
  float safe_altitude = 20.0f;
  float min_altitude = 10.0f;
  float min_clearance = 5.0f;
};

struct LandingConfig
{
  size_t ransac_iterations = 100;
  float ransac_distance_threshold = 0.02f;
  size_t min_points_for_plane = 100;
  
  float max_tilt_angle = 5.0f;
  float min_area = 0.25f;
  float max_roughness = 0.02f;
};

struct VkrConfig
{
  TerrainConfig terrain;
  GeometryConfig geometry;
  PlanningConfig planning;
  LandingConfig landing;
};

}

#endif