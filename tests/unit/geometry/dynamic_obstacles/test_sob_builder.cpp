#include <gtest/gtest.h>
#include "vkr/geometry/dynamic_obstacles/sob_builder.hpp"
#include "vkr/geometry/dynamic_obstacles/swept_obb_bvh.hpp"
#include "vkr/geometry/intersections.hpp"
#include "vkr/config.hpp"
#include <fstream>
#include <filesystem>

namespace vkr 
{
namespace geometry 
{
namespace dynamic_obstacles 
{

class SweptOBBBVHBuilderTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.max_faces_per_leaf = 4;
    config_.bvh_max_leaf_size = 4;
    config_.bvh_sah_traversal_cost = 1.0f;
    config_.bvh_sah_intersection_cost = 1.5f;
    builder_ = std::make_unique<SweptOBBBVHBuilder>(config_);
    
    test_dir_ = std::filesystem::temp_directory_path() / "vkr_sob_test";
    std::filesystem::create_directories(test_dir_);
  }
  
  void TearDown() override 
  {
    std::filesystem::remove_all(test_dir_);
  }
  
  void createTestOBJFile(const std::string& filename) 
  {
    std::ofstream file(test_dir_ / filename);
    file << "# Test cube\n";
    file << "v -1 -1 -1\n";
    file << "v  1 -1 -1\n";
    file << "v  1  1 -1\n";
    file << "v -1  1 -1\n";
    file << "v -1 -1  1\n";
    file << "v  1 -1  1\n";
    file << "v  1  1  1\n";
    file << "v -1  1  1\n";
    file << "f 1 2 3 4\n";
    file << "f 5 8 7 6\n";
    file << "f 1 5 6 2\n";
    file << "f 2 6 7 3\n";
    file << "f 3 7 8 4\n";
    file << "f 1 4 8 5\n";
  }
  
  GeometryConfig config_;
  std::unique_ptr<SweptOBBBVHBuilder> builder_;
  std::filesystem::path test_dir_;
};

TEST_F(SweptOBBBVHBuilderTest, BasicConstruction) 
{
  EXPECT_NE(builder_, nullptr);
  EXPECT_EQ(builder_->getObstacleCount(), 0);
}

TEST_F(SweptOBBBVHBuilderTest, AddFromMesh_ValidOBJ) 
{
  createTestOBJFile("cube.obj");
  
  auto now = std::chrono::steady_clock::now();
  std::vector<Eigen::Vector3f> pos_traj = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(10, 0, 0)
  };
  std::vector<Eigen::Vector3f> rot_traj = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0, 0, M_PI)
  };
  
  builder_->addFromMesh(
    (test_dir_ / "cube.obj").string(),
    123,
    pos_traj,
    rot_traj,
    now,
    now + std::chrono::seconds(10)
  );
  
  EXPECT_EQ(builder_->getObstacleCount(), 1);
}

TEST_F(SweptOBBBVHBuilderTest, AddFromMesh_InvalidFile) 
{
  auto now = std::chrono::steady_clock::now();
  std::vector<Eigen::Vector3f> pos_traj = {Eigen::Vector3f(0, 0, 0)};
  std::vector<Eigen::Vector3f> rot_traj = {Eigen::Vector3f(0, 0, 0)};
  
  builder_->addFromMesh(
    "nonexistent.obj",
    123,
    pos_traj,
    rot_traj,
    now,
    now + std::chrono::seconds(10)
  );
  
  EXPECT_EQ(builder_->getObstacleCount(), 0);
}

TEST_F(SweptOBBBVHBuilderTest, AddWithTimeMapping) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 0.5f, 0.25f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  auto now = std::chrono::steady_clock::now();
  
  builder_->addWithTimeMapping(
    obb,
    789,
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(5, 0, 0)},
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, M_PI/2)},
    {0.0f, 0.1f},
    now,
    now + std::chrono::seconds(20)
  );
  
  EXPECT_EQ(builder_->getObstacleCount(), 1);
}

TEST_F(SweptOBBBVHBuilderTest, SetTrajectoryDegree) 
{
  builder_->setTrajectoryDegree(0);
  builder_->setTrajectoryDegree(10);
  
  EXPECT_EQ(builder_->getObstacleCount(), 0);
}

TEST_F(SweptOBBBVHBuilderTest, Clear) 
{
  auto now = std::chrono::steady_clock::now();
  for (int i = 0; i < 5; ++i) 
  {
    OBB obb;
    obb.center = Eigen::Vector3f(i, 0, 0);
    obb.half_extents = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {Eigen::Vector3f(i, 0, 0)},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      now,
      now + std::chrono::seconds(1)
    );
  }
  
  EXPECT_EQ(builder_->getObstacleCount(), 5);
  
  builder_->clear();
  EXPECT_EQ(builder_->getObstacleCount(), 0);
}

TEST_F(SweptOBBBVHBuilderTest, Build_Empty) 
{
  auto bvh = builder_->build();
  EXPECT_NE(bvh, nullptr);
  
  auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), std::chrono::steady_clock::now());
  EXPECT_FALSE(result.hit);
}

TEST_F(SweptOBBBVHBuilderTest, Build_SingleOBB) 
{
  auto now = std::chrono::steady_clock::now();
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 1, 1);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0)},
    {},
    now,
    now + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  EXPECT_NE(bvh, nullptr);
  
  auto result = bvh->queryPoint(Eigen::Vector3f(0.5f, 0.5f, 0.5f), now);
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_id, 1);
}

TEST_F(SweptOBBBVHBuilderTest, Build_MultipleOBBs) 
{
  auto now = std::chrono::steady_clock::now();
  
  config_.bvh_max_leaf_size = 2;
  builder_ = std::make_unique<SweptOBBBVHBuilder>(config_);
  
  for (int i = 0; i < 10; ++i) 
  {
    OBB obb;
    obb.center = Eigen::Vector3f(i * 3, 0, 0);
    obb.half_extents = Eigen::Vector3f(1, 1, 1);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {Eigen::Vector3f(i * 3, 0, 0)},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      now,
      now + std::chrono::seconds(10)
    );
  }
  
  auto bvh = builder_->build();
  auto stats = bvh->getStats();
  
  EXPECT_EQ(stats.num_objects, 10);
  EXPECT_GT(stats.num_nodes, 10);
  EXPECT_GT(stats.tree_depth, 2);
}

TEST_F(SweptOBBBVHBuilderTest, Build_MovingOBBs)
{
  GeometryConfig config;
  SweptOBBBVHBuilder builder(config);
  
  auto now = std::chrono::steady_clock::now();
  
  OBB obb1;
  obb1.center = Eigen::Vector3f::Zero();
  obb1.half_extents = Eigen::Vector3f(1, 1, 1);
  obb1.orientation = Eigen::Matrix3f::Identity();
  
  builder.addWithTimeMapping(
    obb1, 1,
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0)}, 
    {}, {}, now, now + std::chrono::seconds(1)
  );
  
  auto bvh = builder.build();
  ASSERT_NE(bvh, nullptr);
  
  OBB query_obb;
  query_obb.center = Eigen::Vector3f(5, 0, 0);
  query_obb.half_extents = Eigen::Vector3f(0.5, 0.5, 0.5);
  query_obb.orientation = Eigen::Matrix3f::Identity();
  
  auto time_of_check = now + std::chrono::milliseconds(500);
  auto result = bvh->queryOBB(query_obb, time_of_check, time_of_check);
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_id, 1);
}

TEST_F(SweptOBBBVHBuilderTest, ComputeOptimalOBB_EmptyPoints) 
{
  std::vector<Eigen::Vector3f> points;
  auto obb = computeOptimalOBB(points);
  
  EXPECT_TRUE(obb.center.allFinite());
  EXPECT_TRUE(obb.half_extents.allFinite());
  EXPECT_GT(obb.half_extents.minCoeff(), 0.0f);
}

TEST_F(SweptOBBBVHBuilderTest, ComputeOptimalOBB_SinglePoint) 
{
  std::vector<Eigen::Vector3f> points = {Eigen::Vector3f(1, 2, 3)};
  auto obb = computeOptimalOBB(points);
  
  EXPECT_NEAR(obb.center.x(), 1.0f, 1e-3f);
  EXPECT_NEAR(obb.center.y(), 2.0f, 1e-3f);
  EXPECT_NEAR(obb.center.z(), 3.0f, 1e-3f);
  EXPECT_GT(obb.half_extents.minCoeff(), 0.0f);
}

TEST_F(SweptOBBBVHBuilderTest, ComputeOptimalOBB_AxisAlignedPoints) 
{
  std::vector<Eigen::Vector3f> points = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(2, 0, 0),
    Eigen::Vector3f(0, 1, 0),
    Eigen::Vector3f(2, 1, 0),
    Eigen::Vector3f(0, 0, 0.5f),
    Eigen::Vector3f(2, 0, 0.5f),
    Eigen::Vector3f(0, 1, 0.5f),
    Eigen::Vector3f(2, 1, 0.5f)
  };
  
  auto obb = computeOptimalOBB(points);
  
  EXPECT_NEAR(obb.center.x(), 1.0f, 1e-2f);
  EXPECT_NEAR(obb.center.y(), 0.5f, 1e-2f);
  EXPECT_NEAR(obb.center.z(), 0.25f, 1e-2f);
  
  std::vector<float> extents = {obb.half_extents.x(), obb.half_extents.y(), obb.half_extents.z()};
  std::sort(extents.begin(), extents.end());
  
  EXPECT_NEAR(extents[0], 0.25f, 1e-2f);
  EXPECT_NEAR(extents[1], 0.5f, 1e-2f);
  EXPECT_NEAR(extents[2], 1.0f, 1e-2f);
}

TEST_F(SweptOBBBVHBuilderTest, ComputeOptimalOBB_RotatedPoints) 
{
  float angle = M_PI / 4;
  std::vector<Eigen::Vector3f> points;
  
  for (float x : {-2.0f, 2.0f}) 
  {
    for (float y : {-0.5f, 0.5f}) 
    {
      Eigen::Vector3f p(x, y, 0);
      Eigen::Vector3f rotated(
        p.x() * cos(angle) - p.y() * sin(angle),
        p.x() * sin(angle) + p.y() * cos(angle),
        0
      );
      points.push_back(rotated);
    }
  }
  
  auto obb = computeOptimalOBB(points);
  
  for (const auto& p : points) 
  {
    EXPECT_TRUE(intersectPointOBB(p, obb));
  }
  
  std::vector<float> extents = {obb.half_extents.x(), obb.half_extents.y(), obb.half_extents.z()};
  std::sort(extents.begin(), extents.end());
  
  EXPECT_NEAR(extents[0], 0.0f, 1e-5f);
  EXPECT_NEAR(extents[1], 0.5f, 1e-1f);
  EXPECT_NEAR(extents[2], 2.0f, 1e-1f);
}

TEST_F(SweptOBBBVHBuilderTest, ChainedOperations) 
{
  auto now = std::chrono::steady_clock::now();
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 1, 1);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->setTrajectoryDegree(3)
          .addWithTimeMapping(
            obb,
            1,
            {Eigen::Vector3f(0, 0, 0)},
            {Eigen::Vector3f(0, 0, 0)},
            {},
            now,
            now + std::chrono::seconds(5)
          )
          .addWithTimeMapping(
            obb,
            2,
            {Eigen::Vector3f(5, 0, 0)},
            {Eigen::Vector3f(0, 0, 0)},
            {},
            now,
            now + std::chrono::seconds(5)
          );
  
  EXPECT_EQ(builder_->getObstacleCount(), 2);
  
  auto bvh = builder_->build();
  EXPECT_NE(bvh, nullptr);
  
  EXPECT_EQ(builder_->getObstacleCount(), 0);
}

}
}
}