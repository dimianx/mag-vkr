#ifndef VKR_PLANNING_MULTI_RESOLUTION_GRID_HPP_
#define VKR_PLANNING_MULTI_RESOLUTION_GRID_HPP_

#include "vkr/planning/grid_node.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <vector>
#include <unordered_map>

namespace vkr
{
namespace planning
{

class MultiResolutionGrid
{
public:
  explicit MultiResolutionGrid(const PlanningConfig& config);
  ~MultiResolutionGrid();
  
  enum class Level : uint8_t
  {
    COARSE = 0,
    FINE = 1,
    ULTRA = 2
  };
  
  void initialize(const geometry::BoundingBox& bounds);
  
  NodeId getNode(const Eigen::Vector3f& position, Level level) const;
  
  Eigen::Vector3f getNodePosition(NodeId node_id, Level level) const;
  
  std::vector<NodeId> getNeighbors(NodeId node_id, Level level) const;
  
  float getEdgeCost(NodeId from, NodeId to, Level level) const;
  
  void updateEdgeCost(NodeId from, NodeId to, float new_cost, Level level);
  
  NodeId refineNode(NodeId coarse_node, Level from_level, Level to_level) const;
  NodeId coarsenNode(NodeId fine_node, Level from_level, Level to_level) const;
  
  bool isValidNode(NodeId node_id, Level level) const;
  
  float getResolution(Level level) const;
  
  void batchGetNeighbors(const NodeId* nodes, size_t count, Level level,
                        std::vector<std::vector<NodeId>>& neighbors) const;
  
  void clearLevel(Level level);
  size_t getMemoryUsage() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct GridLevel
  {
    float resolution;
    Eigen::Vector3i dimensions;
    geometry::BoundingBox bounds;
    std::unordered_map<uint64_t, GridNode> nodes;
    
    struct Block
    {
      static constexpr size_t SIZE = 16;
      std::vector<GridNode> nodes;
    };
    std::unordered_map<uint64_t, std::unique_ptr<Block>> blocks;
  };
  
  std::array<GridLevel, 3> levels_;
  PlanningConfig config_;
  
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  uint64_t positionToMorton(const Eigen::Vector3f& pos, Level level) const;
  Eigen::Vector3f mortonToPosition(uint64_t morton, Level level) const;
  Eigen::Vector3i positionToGrid(const Eigen::Vector3f& pos, Level level) const;
};

}
}

#endif