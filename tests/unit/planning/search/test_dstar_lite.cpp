#include <gtest/gtest.h>
#include "vkr/planning/search/dstar_lite.hpp"
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/geometry/primitives.hpp"
#include <memory>
#include <thread>
#include <chrono>

namespace vkr 
{
namespace planning 
{
namespace search 
{

class DStarLiteTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.coarse_grid_size = 20.0f;
    config_.fine_grid_size = 5.0f;
    config_.max_runtime_ms = 100.0f;
    
    grid_ = std::make_shared<MultiResolutionGrid>(config_);
    
    geometry::BoundingBox bounds;
    bounds.min = Eigen::Vector3f(0, 0, 0);
    bounds.max = Eigen::Vector3f(500, 500, 100);
    grid_->initialize(bounds);
    
    dstar_ = std::make_unique<DStarLite>(config_);
  }
  
  PlanningConfig config_;
  std::shared_ptr<MultiResolutionGrid> grid_;
  std::unique_ptr<DStarLite> dstar_;
};

TEST_F(DStarLiteTest, Initialization) 
{
  NodeId start = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(450, 450, 50), MultiResolutionGrid::Level::FINE);
  
  ASSERT_NE(start, INVALID_NODE);
  ASSERT_NE(goal, INVALID_NODE);
  
  dstar_->initialize(start, goal, grid_);
  
  auto path = dstar_->getCurrentPath();
  EXPECT_FALSE(path.empty());
  
  EXPECT_EQ(path.front(), start);
  EXPECT_EQ(path.back(), goal);
}

TEST_F(DStarLiteTest, EdgeCostUpdate) 
{
  NodeId start = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(250, 250, 50), MultiResolutionGrid::Level::FINE);
  
  dstar_->initialize(start, goal, grid_);
  
  auto initial_path = dstar_->getCurrentPath();
  size_t initial_length = initial_path.size();
  
  if (initial_path.size() > 2) 
  {
    DStarLite::EdgeUpdate update;
    update.from = initial_path[1];
    update.to = initial_path[2];
    update.new_cost = INFINITY_F;
    
    dstar_->updateEdgeCost(update);
    
    EXPECT_TRUE(dstar_->needsReplan());
    
    bool success = dstar_->replan();
    EXPECT_TRUE(success);
    
    auto new_path = dstar_->getCurrentPath();
    EXPECT_FALSE(new_path.empty());
    
    bool contains_blocked = false;
    for (size_t i = 0; i < new_path.size() - 1; ++i) 
    {
      if (new_path[i] == initial_path[1] && new_path[i+1] == initial_path[2]) 
      {
        contains_blocked = true;
        break;
      }
    }
    EXPECT_FALSE(contains_blocked);
  }
}

TEST_F(DStarLiteTest, MovingStart) 
{
  NodeId start1 = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(450, 450, 50), MultiResolutionGrid::Level::FINE);
  
  dstar_->initialize(start1, goal, grid_);
  
  auto path1 = dstar_->getCurrentPath();
  
  NodeId start2 = grid_->getNode(Eigen::Vector3f(100, 100, 50), MultiResolutionGrid::Level::FINE);
  dstar_->updateStart(start2);
  
  auto path2 = dstar_->getCurrentPath();
  
  EXPECT_EQ(path2.front(), start2);
  EXPECT_EQ(path2.back(), goal);
  
  NodeId start3 = grid_->getNode(Eigen::Vector3f(150, 150, 50), MultiResolutionGrid::Level::FINE);
  dstar_->updateStart(start3);
  
  auto path3 = dstar_->getCurrentPath();
  EXPECT_EQ(path3.front(), start3);
}

TEST_F(DStarLiteTest, Statistics) 
{
  NodeId start = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(250, 250, 50), MultiResolutionGrid::Level::FINE);
  
  dstar_->initialize(start, goal, grid_);
  
  DStarLite::EdgeUpdate update;
  update.from = start;
  update.to = grid_->getNeighbors(start, MultiResolutionGrid::Level::FINE).front();
  update.new_cost = 10.0f;
  
  dstar_->updateEdgeCost(update);
  dstar_->replan();
  
  auto stats = dstar_->getStats();
  
  EXPECT_GT(stats.nodes_updated, 0u);
  EXPECT_GT(stats.replan_iterations, 0u);
  EXPECT_GT(stats.replan_time_ms, 0.0f);
}

TEST_F(DStarLiteTest, RapidUpdates) 
{
  NodeId start = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(250, 250, 50), MultiResolutionGrid::Level::FINE);
  
  dstar_->initialize(start, goal, grid_);
  
  for (int i = 0; i < 5; ++i) 
  {
    DStarLite::EdgeUpdate update;
    NodeId node = grid_->getNode(Eigen::Vector3f(100 + i * 20, 100, 50), 
                                 MultiResolutionGrid::Level::FINE);
    
    if (node != INVALID_NODE) 
    {
      auto neighbors = grid_->getNeighbors(node, MultiResolutionGrid::Level::FINE);
      if (!neighbors.empty()) 
      {
        update.from = node;
        update.to = neighbors.front();
        update.new_cost = 5.0f + i;
        
        dstar_->updateEdgeCost(update);
      }
    }
  }
  
  if (dstar_->needsReplan()) 
  {
    bool success = dstar_->replan();
    EXPECT_TRUE(success);
  }
}

TEST_F(DStarLiteTest, JSONExport) 
{
  NodeId start = grid_->getNode(Eigen::Vector3f(50, 50, 50), MultiResolutionGrid::Level::FINE);
  NodeId goal = grid_->getNode(Eigen::Vector3f(250, 250, 50), MultiResolutionGrid::Level::FINE);
  
  dstar_->initialize(start, goal, grid_);
  
  std::string filename = "dstar_lite_test.json";
  dstar_->exportToJSON(filename);
}

}
}
}