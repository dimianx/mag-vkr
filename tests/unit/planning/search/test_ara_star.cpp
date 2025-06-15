#include <gtest/gtest.h>
#include "vkr/planning/search/ara_star.hpp"
#include "vkr/planning/multi_resolution_grid.hpp"
#include "vkr/geometry/primitives.hpp"
#include <memory>
#include <chrono>

namespace vkr 
{
namespace planning 
{
namespace search 
{

class ARAStarTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.coarse_grid_size = 20.0f;
    config_.fine_grid_size = 5.0f;
    config_.initial_epsilon = 2.5f;
    config_.max_runtime_ms = 100.0f;
    
    grid_ = std::make_shared<MultiResolutionGrid>(config_);
    
    geometry::BoundingBox bounds;
    bounds.min = Eigen::Vector3f(0, 0, 0);
    bounds.max = Eigen::Vector3f(1000, 1000, 200);
    grid_->initialize(bounds);
    
    search_ = std::make_unique<ARAStarSearch>(config_);
    search_->setGrid(grid_);
  }
  
  PlanningConfig config_;
  std::shared_ptr<MultiResolutionGrid> grid_;
  std::unique_ptr<ARAStarSearch> search_;
};

TEST_F(ARAStarTest, BasicSearch) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(100, 100, 50);
  request.goal = Eigen::Vector3f(900, 900, 50);
  request.max_runtime_ms = 100.0f;
  
  auto result = search_->search(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  EXPECT_FALSE(result.path.empty());
  EXPECT_GT(result.nodes_expanded, 0u);
  EXPECT_GT(result.nodes_generated, 0u);
  
  EXPECT_LT((result.path.front() - request.start).norm(), config_.fine_grid_size);
  EXPECT_LT((result.path.back() - request.goal).norm(), config_.fine_grid_size);
}

TEST_F(ARAStarTest, HeuristicWeights) 
{
  search_->setHeuristicWeights(1.0f, 0.5f, 0.2f);
  
  PlanRequest request;
  request.start = Eigen::Vector3f(100, 100, 50);
  request.goal = Eigen::Vector3f(500, 500, 50);
  
  auto result = search_->search(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  search_->setHeuristicWeights(2.0f, 0.0f, 0.0f);
  
  auto result2 = search_->search(request);
  EXPECT_EQ(result2.status, Status::SUCCESS);
}

TEST_F(ARAStarTest, ValidityFunction) 
{
  search_->setValidityFunction([](const Eigen::Vector3f& pos) 
  {
    if ((pos.head<2>() - Eigen::Vector2f(500, 500)).norm() < 100) 
    {
      return false;
    }
    return true;
  });
  
  PlanRequest request;
  request.start = Eigen::Vector3f(100, 100, 50);
  request.goal = Eigen::Vector3f(900, 900, 50);
  
  auto result = search_->search(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  for (const auto& pos : result.path) 
  {
    float dist = (pos.head<2>() - Eigen::Vector2f(500, 500)).norm();
    EXPECT_GE(dist, 100.0f);
  }
}

TEST_F(ARAStarTest, AnytimeImprovement) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(100, 100, 50);
  request.goal = Eigen::Vector3f(900, 900, 50);
  request.max_runtime_ms = 50.0f;
  
  auto initial = search_->search(request);
  EXPECT_EQ(initial.status, Status::SUCCESS);
  
  float initial_cost = initial.cost;
  float initial_epsilon = initial.epsilon;
  
  bool improved = search_->improveSolution(1.5f, 50.0f);
  
  if (improved) 
  {
    auto better = search_->getCurrentSolution();
    EXPECT_LE(better.cost, initial_cost);
    EXPECT_LT(better.epsilon, initial_epsilon);
  }
}

TEST_F(ARAStarTest, MultiResolutionSearch) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(50, 50, 50);
  request.goal = Eigen::Vector3f(950, 950, 50);
  request.max_runtime_ms = 200.0f;
  
  auto result = search_->search(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  auto stats = search_->getStats();
  EXPECT_GT(stats.nodes_generated, 0u);
}

TEST_F(ARAStarTest, TimeoutBehavior) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(50, 50, 50);
  request.goal = Eigen::Vector3f(950, 950, 50);
  request.max_runtime_ms = 0.001f;
  
  auto result = search_->search(request);
  
  if (result.status == Status::SUCCESS) 
  {
    EXPECT_GT(result.epsilon, 1.0f);
  } 
  else 
  {
    EXPECT_EQ(result.status, Status::TIMEOUT);
  }
}

TEST_F(ARAStarTest, Statistics) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(100, 100, 50);
  request.goal = Eigen::Vector3f(300, 300, 50);
  
  search_->search(request);
  
  auto stats = search_->getStats();
  
  EXPECT_GT(stats.nodes_expanded, 0u);
  EXPECT_GE(stats.nodes_generated, stats.nodes_expanded);
  EXPECT_EQ(stats.current_epsilon, config_.initial_epsilon);
  EXPECT_GT(stats.solution_cost, 0.0f);
}

}
}
}