#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/geometry/morton.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <unordered_map>

namespace vkr
{
namespace planning
{

namespace
{
  
const std::array<Eigen::Vector3i, 26> NEIGHBOR_OFFSETS = 
{
  {
    {
      -1, 0, 0
    }, 
    {
      1, 0, 0
    }, 
    {
      0, -1, 0
    }, 
    {
      0, 1, 0
    }, 
    {
      0, 0, -1
    }, 
    {
      0, 0, 1
    },
    {
      -1, -1, 0
    }, 
    {
      -1, 1, 0
    }, 
    {
      1, -1, 0
    }, 
    {
      1, 1, 0
    },
    {
      -1, 0, -1
    }, 
    {
      -1, 0, 1
    }, 
    {
      1, 0, -1
    }, 
    {
      1, 0, 1
    },
    {
      0, -1, -1
    }, 
    {
      0, -1, 1
    }, 
    {
      0, 1, -1
    }, 
    {
      0, 1, 1
    },
    {
      -1, -1, -1
    }, 
    {
      -1, -1, 1
    }, 
    {
      -1, 1, -1
    }, 
    {
      -1, 1, 1
    },
    {
      1, -1, -1
    }, 
    {
      1, -1, 1
    }, 
    {
      1, 1, -1
    }, 
    {
      1, 1, 1
    }
  }
};

const std::array<float, 26> NEIGHBOR_DISTANCES = 
{
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.414f, 1.414f, 1.414f, 1.414f, 1.414f, 1.414f, 
  1.414f, 1.414f, 1.414f, 1.414f, 1.414f, 1.414f,
  1.732f, 1.732f, 1.732f, 1.732f, 1.732f, 1.732f, 1.732f, 1.732f
};

}

struct MultiResolutionGrid::Implementation 
{
  PlanningConfig config;
  std::array<GridLevel, 3> levels;
  mutable std::array<std::mutex, 3> level_mutexes;
  
  static constexpr uint32_t LEVEL_BITS = 2;
  static constexpr uint32_t LEVEL_SHIFT = 30;
  static constexpr uint32_t MORTON_MASK = (1u << LEVEL_SHIFT) - 1;
  
  struct EdgeKey 
  {
    uint64_t from;
    uint64_t to;
    
    bool operator==(const EdgeKey& other) const 
    {
      return from == other.from && to == other.to;
    }
  };
  
  struct EdgeKeyHash 
  {
    size_t operator()(const EdgeKey& key) const 
    {
      return std::hash<uint64_t>
      {
      }(key.from) ^ (std::hash<uint64_t>
      {
      }(key.to) << 1);
    }
  };
  
  std::array<std::unordered_map<EdgeKey, float, EdgeKeyHash>, 3> edge_costs;
  mutable std::array<std::mutex, 3> edge_cost_mutexes;
  
  NodeId encodeNodeId(uint64_t morton, Level level) const 
  {
    uint32_t level_bits = static_cast<uint32_t>(level);
    uint32_t morton_bits = static_cast<uint32_t>(morton & MORTON_MASK);
    return (level_bits << LEVEL_SHIFT) | morton_bits;
  }
  
  void decodeNodeId(NodeId node_id, uint64_t& morton, Level& level) const 
  {
    level = static_cast<Level>(node_id >> LEVEL_SHIFT);
    morton = node_id & MORTON_MASK;
  }
  
  GridNode* getOrCreateNode(uint64_t morton, Level level) 
  {
    auto& grid_level = levels[static_cast<size_t>(level)];
    
    auto it = grid_level.nodes.find(morton);
    if (it != grid_level.nodes.end()) 
    {
      return &it->second;
    }
    
    uint64_t block_id = morton / GridLevel::Block::SIZE;
    uint64_t block_offset = morton % GridLevel::Block::SIZE;
    
    auto block_it = grid_level.blocks.find(block_id);
    if (block_it == grid_level.blocks.end()) 
    {
      auto new_block = std::make_unique<GridLevel::Block>();
      new_block->nodes.resize(GridLevel::Block::SIZE);
      block_it = grid_level.blocks.emplace(block_id, std::move(new_block)).first;
    }
    
    GridNode& node = block_it->second->nodes[block_offset];
    node.morton_code = morton;
    node.level = static_cast<uint8_t>(level);
    
    grid_level.nodes[morton] = node;
    
    return &grid_level.nodes[morton];
  }
};

MultiResolutionGrid::MultiResolutionGrid(const PlanningConfig& config)
  : config_(config), impl_(std::make_unique<Implementation>())
{
  impl_->config = config;
  
  impl_->levels[0].resolution = config.coarse_grid_size;
  impl_->levels[1].resolution = config.fine_grid_size;
  impl_->levels[2].resolution = config.fine_grid_size / 4.0f;
}

MultiResolutionGrid::~MultiResolutionGrid() = default;

void MultiResolutionGrid::initialize(const geometry::BoundingBox& bounds) 
{
  for (size_t i = 0; i < impl_->levels.size(); ++i) 
  {
    auto& level = impl_->levels[i];
    level.bounds = bounds;
    
    Eigen::Vector3f size = bounds.max - bounds.min;
    level.dimensions = ((size / level.resolution).array().template cast<int>() + 1).matrix();
    
    level.nodes.clear();
    level.blocks.clear();
    
    spdlog::info("MultiResolutionGrid: Level {} initialized with resolution {} and dimensions {}x{}x{}", 
                 i, level.resolution, level.dimensions.x(), level.dimensions.y(), level.dimensions.z());
  }
}

NodeId MultiResolutionGrid::getNode(const Eigen::Vector3f& position, Level level) const 
{
  const auto& grid_level = impl_->levels[static_cast<size_t>(level)];
  
  if (!grid_level.bounds.contains(position)) 
  {
    return INVALID_NODE;
  }
  
  Eigen::Vector3i grid_pos = positionToGrid(position, level);
  
  uint64_t morton = geometry::morton3d::encode(
    static_cast<uint32_t>(grid_pos.x()),
    static_cast<uint32_t>(grid_pos.y()),
    static_cast<uint32_t>(grid_pos.z())
  );
  
  return impl_->encodeNodeId(morton, level);
}

Eigen::Vector3f MultiResolutionGrid::getNodePosition(NodeId node_id, Level level) const 
{
  if (node_id == INVALID_NODE) 
  {
    return Eigen::Vector3f::Zero();
  }
  
  uint64_t morton;
  Level decoded_level;
  impl_->decodeNodeId(node_id, morton, decoded_level);
  
  if (decoded_level == level) 
  {
    return mortonToPosition(morton, decoded_level);
  }
  
  Eigen::Vector3f pos = mortonToPosition(morton, decoded_level);
  
  const auto& target_level = impl_->levels[static_cast<size_t>(level)];
  Eigen::Vector3f snapped_pos;
  snapped_pos.x() = std::round(pos.x() / target_level.resolution) * target_level.resolution;
  snapped_pos.y() = std::round(pos.y() / target_level.resolution) * target_level.resolution;
  snapped_pos.z() = std::round(pos.z() / target_level.resolution) * target_level.resolution;
  
  return snapped_pos;
}

std::vector<NodeId> MultiResolutionGrid::getNeighbors(NodeId node_id, Level level) const 
{
  uint64_t morton;
  Level decoded_level;
  impl_->decodeNodeId(node_id, morton, decoded_level);
  
  if (decoded_level != level) 
  {
    return 
    {
    };
  }
  
  uint32_t x, y, z;
  geometry::morton3d::decode(morton, x, y, z);
  
  const auto& grid_level = impl_->levels[static_cast<size_t>(level)];
  std::vector<NodeId> neighbors;
  neighbors.reserve(26);
  
  Eigen::Vector3f node_pos = mortonToPosition(morton, level);
  float max_neighbor_dist = grid_level.resolution * std::sqrt(3.0f) * 1.01f;
  
  for (size_t i = 0; i < NEIGHBOR_OFFSETS.size(); ++i) 
  {
    const auto& offset = NEIGHBOR_OFFSETS[i];
    int nx = static_cast<int>(x) + offset.x();
    int ny = static_cast<int>(y) + offset.y();
    int nz = static_cast<int>(z) + offset.z();
    
    if (nx >= 0 && nx < grid_level.dimensions.x() &&
        ny >= 0 && ny < grid_level.dimensions.y() &&
        nz >= 0 && nz < grid_level.dimensions.z()) 
    {
      uint64_t neighbor_morton = geometry::morton3d::encode(
        static_cast<uint32_t>(nx),
        static_cast<uint32_t>(ny),
        static_cast<uint32_t>(nz)
      );
      
      Eigen::Vector3f neighbor_pos = mortonToPosition(neighbor_morton, level);
      float dist = (neighbor_pos - node_pos).norm();
      
      if (dist <= max_neighbor_dist) 
      {
        neighbors.push_back(impl_->encodeNodeId(neighbor_morton, level));
      }
    }
  }
  
  return neighbors;
}

float MultiResolutionGrid::getEdgeCost(NodeId from, NodeId to, Level level) const 
{
  uint64_t morton_from, morton_to;
  Level level_from, level_to;
  impl_->decodeNodeId(from, morton_from, level_from);
  impl_->decodeNodeId(to, morton_to, level_to);
  
  if (level_from != level || level_to != level) 
  {
    return INFINITY_F;
  }
  
  {
    std::lock_guard<std::mutex> lock(impl_->edge_cost_mutexes[static_cast<size_t>(level)]);
    Implementation::EdgeKey key
    {
      morton_from, morton_to
    };
    auto it = impl_->edge_costs[static_cast<size_t>(level)].find(key);
    if (it != impl_->edge_costs[static_cast<size_t>(level)].end()) 
    {
      return it->second;
    }
  }
  
  uint32_t x1, y1, z1, x2, y2, z2;
  geometry::morton3d::decode(morton_from, x1, y1, z1);
  geometry::morton3d::decode(morton_to, x2, y2, z2);
  
  int dx = static_cast<int>(x2) - static_cast<int>(x1);
  int dy = static_cast<int>(y2) - static_cast<int>(y1);
  int dz = static_cast<int>(z2) - static_cast<int>(z1);
  
  for (size_t i = 0; i < NEIGHBOR_OFFSETS.size(); ++i) 
  {
    if (NEIGHBOR_OFFSETS[i].x() == dx && 
        NEIGHBOR_OFFSETS[i].y() == dy && 
        NEIGHBOR_OFFSETS[i].z() == dz) 
    {
      float base_cost = NEIGHBOR_DISTANCES[i] * impl_->levels[static_cast<size_t>(level)].resolution;
      return base_cost;
    }
  }
  
  return INFINITY_F;
}

void MultiResolutionGrid::updateEdgeCost(NodeId from, NodeId to, float new_cost, Level level) 
{
  uint64_t morton_from, morton_to;
  Level level_from, level_to;
  impl_->decodeNodeId(from, morton_from, level_from);
  impl_->decodeNodeId(to, morton_to, level_to);
  
  if (level_from != level || level_to != level) 
  {
    return;
  }
  
  std::lock_guard<std::mutex> lock(impl_->edge_cost_mutexes[static_cast<size_t>(level)]);
  
  Implementation::EdgeKey key_forward
  {
    morton_from, morton_to
  };
  impl_->edge_costs[static_cast<size_t>(level)][key_forward] = new_cost;
  
  Implementation::EdgeKey key_backward
  {
    morton_to, morton_from
  };
  impl_->edge_costs[static_cast<size_t>(level)][key_backward] = new_cost;
}

NodeId MultiResolutionGrid::refineNode(NodeId coarse_node, Level from_level, Level to_level) const 
{
  if (from_level >= to_level) 
  {
    return coarse_node;
  }
  
  Eigen::Vector3f pos = getNodePosition(coarse_node, from_level);
  
  NodeId fine_node = getNode(pos, to_level);
  
  if (static_cast<int>(to_level) - static_cast<int>(from_level) > 1) 
  {
    for (int level = static_cast<int>(from_level) + 1; level < static_cast<int>(to_level); ++level) 
    {
      NodeId intermediate = getNode(pos, static_cast<Level>(level));
      if (intermediate == INVALID_NODE) 
      {
        return INVALID_NODE;
      }
    }
  }
  
  return fine_node;
}

NodeId MultiResolutionGrid::coarsenNode(NodeId fine_node, Level from_level, Level to_level) const 
{
  if (from_level <= to_level) 
  {
    return fine_node;
  }
  
  Eigen::Vector3f pos = getNodePosition(fine_node, from_level);
  
  const auto& coarse_level = impl_->levels[static_cast<size_t>(to_level)];
  Eigen::Vector3f snapped_pos;
  snapped_pos.x() = std::round(pos.x() / coarse_level.resolution) * coarse_level.resolution;
  snapped_pos.y() = std::round(pos.y() / coarse_level.resolution) * coarse_level.resolution;
  snapped_pos.z() = std::round(pos.z() / coarse_level.resolution) * coarse_level.resolution;
  
  return getNode(snapped_pos, to_level);
}

bool MultiResolutionGrid::isValidNode(NodeId node_id, Level level) const 
{
  if (node_id == INVALID_NODE) 
  {
    return false;
  }
  
  uint64_t morton;
  Level decoded_level;
  impl_->decodeNodeId(node_id, morton, decoded_level);
  
  return decoded_level == level;
}

float MultiResolutionGrid::getResolution(Level level) const 
{
  return impl_->levels[static_cast<size_t>(level)].resolution;
}

void MultiResolutionGrid::batchGetNeighbors(const NodeId* nodes, size_t count, Level level,
                                           std::vector<std::vector<NodeId>>& neighbors) const 
{
  neighbors.resize(count);
  
  tbb::parallel_for(tbb::blocked_range<size_t>(0, count),
    [&](const tbb::blocked_range<size_t>& range) 
    {
      for (size_t i = range.begin(); i != range.end(); ++i) 
      {
        neighbors[i] = getNeighbors(nodes[i], level);
      }
    });
}

void MultiResolutionGrid::clearLevel(Level level) 
{
  std::lock_guard<std::mutex> lock(impl_->level_mutexes[static_cast<size_t>(level)]);
  
  auto& grid_level = impl_->levels[static_cast<size_t>(level)];
  grid_level.nodes.clear();
  grid_level.blocks.clear();
}

size_t MultiResolutionGrid::getMemoryUsage() const 
{
  size_t total = sizeof(*this) + sizeof(*impl_);
  
  for (size_t i = 0; i < impl_->levels.size(); ++i) 
  {
    const auto& level = impl_->levels[i];
    
    total += level.nodes.size() * (sizeof(uint64_t) + sizeof(GridNode));
    total += level.nodes.bucket_count() * sizeof(void*);
    
    total += level.blocks.size() * sizeof(std::pair<uint64_t, std::unique_ptr<GridLevel::Block>>);
    total += level.blocks.bucket_count() * sizeof(void*);
    
    for (const auto& [block_id, block] : level.blocks) 
    {
      if (block) 
      {
        total += sizeof(GridLevel::Block);
        total += block->nodes.capacity() * sizeof(GridNode);
      }
    }
    
    const auto& edge_map = impl_->edge_costs[i];
    total += edge_map.size() * (sizeof(Implementation::EdgeKey) + sizeof(float));
    total += edge_map.bucket_count() * sizeof(void*);
  }
  
  total += impl_->level_mutexes.size() * sizeof(std::mutex);
  total += impl_->edge_cost_mutexes.size() * sizeof(std::mutex);
  
  return total;
}

void MultiResolutionGrid::exportToJSON(const std::string& filename) const 
{
  nlohmann::json j;
  
  j["config"]["coarse_resolution"] = impl_->levels[0].resolution;
  j["config"]["fine_resolution"] = impl_->levels[1].resolution;
  j["config"]["ultra_resolution"] = impl_->levels[2].resolution;
  
  for (size_t i = 0; i < impl_->levels.size(); ++i) 
  {
    const auto& level = impl_->levels[i];
    auto& level_json = j["levels"][i];
    
    level_json["resolution"] = level.resolution;
    level_json["dimensions"] = 
    {
      level.dimensions.x(), level.dimensions.y(), level.dimensions.z()
    };
    level_json["bounds"]["min"] = 
    {
      level.bounds.min.x(), level.bounds.min.y(), level.bounds.min.z()
    };
    level_json["bounds"]["max"] = 
    {
      level.bounds.max.x(), level.bounds.max.y(), level.bounds.max.z()
    };
    level_json["node_count"] = level.nodes.size();
  }
  
  j["memory"]["total_bytes"] = getMemoryUsage();
  
  std::ofstream ofs(filename);
  ofs << j.dump(2);
}

uint64_t MultiResolutionGrid::positionToMorton(const Eigen::Vector3f& pos, Level level) const 
{
  Eigen::Vector3i grid_pos = positionToGrid(pos, level);
  return geometry::morton3d::encode(
    static_cast<uint32_t>(grid_pos.x()),
    static_cast<uint32_t>(grid_pos.y()),
    static_cast<uint32_t>(grid_pos.z())
  );
}

Eigen::Vector3f MultiResolutionGrid::mortonToPosition(uint64_t morton, Level level) const 
{
  uint32_t x, y, z;
  geometry::morton3d::decode(morton, x, y, z);
  
  const auto& grid_level = impl_->levels[static_cast<size_t>(level)];
  
  Eigen::Vector3f pos;
  pos.x() = grid_level.bounds.min.x() + (x + 0.5f) * grid_level.resolution;
  pos.y() = grid_level.bounds.min.y() + (y + 0.5f) * grid_level.resolution;
  pos.z() = grid_level.bounds.min.z() + (z + 0.5f) * grid_level.resolution;
  
  return pos;
}

Eigen::Vector3i MultiResolutionGrid::positionToGrid(const Eigen::Vector3f& pos, Level level) const 
{
  const auto& grid_level = impl_->levels[static_cast<size_t>(level)];
  
  Eigen::Vector3f rel_pos = pos - grid_level.bounds.min;
  Eigen::Vector3i grid_pos = (rel_pos / grid_level.resolution).array().template cast<int>();
  
  grid_pos = grid_pos.cwiseMax(Eigen::Vector3i::Zero());
  grid_pos = grid_pos.cwiseMin(grid_level.dimensions - Eigen::Vector3i::Ones());
  
  return grid_pos;
}

}
}