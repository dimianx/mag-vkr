#include <gtest/gtest.h>
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/geometry/primitives.hpp"
#include <memory>

namespace vkr 
{
namespace planning 
{

class MultiResolutionGridTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.coarse_grid_size = 20.0f;
    config_.fine_grid_size = 5.0f;
    
    grid_ = std::make_unique<MultiResolutionGrid>(config_);
    
    bounds_.min = Eigen::Vector3f(0, 0, 0);
    bounds_.max = Eigen::Vector3f(1000, 1000, 200);
    grid_->initialize(bounds_);
  }
  
  PlanningConfig config_;
  std::unique_ptr<MultiResolutionGrid> grid_;
  geometry::BoundingBox bounds_;
};

TEST_F(MultiResolutionGridTest, Initialization) 
{
  EXPECT_EQ(grid_->getResolution(MultiResolutionGrid::Level::COARSE), 20.0f);
  EXPECT_EQ(grid_->getResolution(MultiResolutionGrid::Level::FINE), 5.0f);
}

TEST_F(MultiResolutionGridTest, NodeCreation) 
{
  Eigen::Vector3f pos(100, 200, 50);
  
  NodeId coarse_node = grid_->getNode(pos, MultiResolutionGrid::Level::COARSE);
  NodeId fine_node = grid_->getNode(pos, MultiResolutionGrid::Level::FINE);
  
  EXPECT_NE(coarse_node, INVALID_NODE);
  EXPECT_NE(fine_node, INVALID_NODE);
  EXPECT_NE(coarse_node, fine_node);
  
  Eigen::Vector3f coarse_pos = grid_->getNodePosition(coarse_node, MultiResolutionGrid::Level::COARSE);
  Eigen::Vector3f fine_pos = grid_->getNodePosition(fine_node, MultiResolutionGrid::Level::FINE);
  
  EXPECT_LE((coarse_pos - pos).norm(), config_.coarse_grid_size * std::sqrt(3.0f));
  EXPECT_LE((fine_pos - pos).norm(), config_.fine_grid_size * std::sqrt(3.0f));
}

TEST_F(MultiResolutionGridTest, OutOfBounds) 
{
  Eigen::Vector3f out_pos(2000, 2000, 500);
  
  NodeId node = grid_->getNode(out_pos, MultiResolutionGrid::Level::FINE);
  EXPECT_EQ(node, INVALID_NODE);
}

TEST_F(MultiResolutionGridTest, Neighbors) 
{
  Eigen::Vector3f pos(500, 500, 100);
  NodeId node = grid_->getNode(pos, MultiResolutionGrid::Level::FINE);
  
  Eigen::Vector3f node_center_pos = grid_->getNodePosition(node, MultiResolutionGrid::Level::FINE);
  
  auto neighbors = grid_->getNeighbors(node, MultiResolutionGrid::Level::FINE);
  
  EXPECT_GT(neighbors.size(), 0u);
  EXPECT_LE(neighbors.size(), 26u);
  
  for (NodeId neighbor : neighbors) 
  {
    EXPECT_NE(neighbor, INVALID_NODE);
    
    Eigen::Vector3f neighbor_pos = grid_->getNodePosition(neighbor, MultiResolutionGrid::Level::FINE);
    float dist = (neighbor_pos - node_center_pos).norm();
    EXPECT_LE(dist, config_.fine_grid_size * std::sqrt(3.0f) * 1.1f);
  }
}

TEST_F(MultiResolutionGridTest, EdgeCosts) 
{
  Eigen::Vector3f pos1(100, 100, 50);
  Eigen::Vector3f pos2(105, 100, 50);
  
  NodeId node1 = grid_->getNode(pos1, MultiResolutionGrid::Level::FINE);
  NodeId node2 = grid_->getNode(pos2, MultiResolutionGrid::Level::FINE);
  
  float cost = grid_->getEdgeCost(node1, node2, MultiResolutionGrid::Level::FINE);
  
  EXPECT_NEAR(cost, 5.0f, 0.1f);
  
  grid_->updateEdgeCost(node1, node2, 10.0f, MultiResolutionGrid::Level::FINE);
  
  float new_cost = grid_->getEdgeCost(node1, node2, MultiResolutionGrid::Level::FINE);
  EXPECT_EQ(new_cost, 10.0f);
}

TEST_F(MultiResolutionGridTest, LevelConversion) 
{
  Eigen::Vector3f pos(100, 100, 50);
  
  NodeId fine_node = grid_->getNode(pos, MultiResolutionGrid::Level::FINE);
  NodeId coarse_from_fine = grid_->coarsenNode(fine_node, 
                                               MultiResolutionGrid::Level::FINE,
                                               MultiResolutionGrid::Level::COARSE);
  
  EXPECT_NE(coarse_from_fine, INVALID_NODE);
  
  Eigen::Vector3f pos2(102, 103, 52);
  NodeId fine_node2 = grid_->getNode(pos2, MultiResolutionGrid::Level::FINE);
  NodeId coarse_from_fine2 = grid_->coarsenNode(fine_node2,
                                                MultiResolutionGrid::Level::FINE,
                                                MultiResolutionGrid::Level::COARSE);
  
  EXPECT_EQ(coarse_from_fine, coarse_from_fine2);
  
  NodeId fine_from_coarse = grid_->refineNode(coarse_from_fine,
                                              MultiResolutionGrid::Level::COARSE,
                                              MultiResolutionGrid::Level::FINE);
  EXPECT_NE(fine_from_coarse, INVALID_NODE);
}

TEST_F(MultiResolutionGridTest, BatchNeighbors) 
{
  constexpr size_t count = 10;
  std::vector<NodeId> nodes;
  
  for (size_t i = 0; i < count; ++i) 
  {
    Eigen::Vector3f pos(100 + i * 50, 100 + i * 50, 50);
    nodes.push_back(grid_->getNode(pos, MultiResolutionGrid::Level::FINE));
  }
  
  std::vector<std::vector<NodeId>> all_neighbors;
  grid_->batchGetNeighbors(nodes.data(), count, MultiResolutionGrid::Level::FINE, all_neighbors);
  
  EXPECT_EQ(all_neighbors.size(), count);
  
  for (const auto& neighbors : all_neighbors) 
  {
    EXPECT_GT(neighbors.size(), 0u);
    EXPECT_LE(neighbors.size(), 26u);
  }
}

TEST_F(MultiResolutionGridTest, NodeValidity) 
{
  Eigen::Vector3f pos(100, 100, 50);
  NodeId node = grid_->getNode(pos, MultiResolutionGrid::Level::FINE);
  
  EXPECT_TRUE(grid_->isValidNode(node, MultiResolutionGrid::Level::FINE));
  EXPECT_FALSE(grid_->isValidNode(INVALID_NODE, MultiResolutionGrid::Level::FINE));
  
  EXPECT_FALSE(grid_->isValidNode(node, MultiResolutionGrid::Level::COARSE));
}

TEST_F(MultiResolutionGridTest, GridResolutions) 
{
  EXPECT_EQ(grid_->getResolution(MultiResolutionGrid::Level::COARSE), config_.coarse_grid_size);
  EXPECT_EQ(grid_->getResolution(MultiResolutionGrid::Level::FINE), config_.fine_grid_size);
  
  EXPECT_EQ(grid_->getResolution(MultiResolutionGrid::Level::ULTRA), config_.fine_grid_size / 4.0f);
}

TEST_F(MultiResolutionGridTest, JSONExport) 
{
  for (int i = 0; i < 10; ++i) 
  {
    Eigen::Vector3f pos(i * 50, i * 50, 50);
    grid_->getNode(pos, MultiResolutionGrid::Level::COARSE);
    grid_->getNode(pos, MultiResolutionGrid::Level::FINE);
  }
  
  std::string filename = "multi_resolution_grid_test.json";
  grid_->exportToJSON(filename);
}

}
}