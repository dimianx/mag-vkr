#include <gtest/gtest.h>
#include "vkr/geometry/dynamic_obstacles/swept_obb_bvh.hpp"
#include "vkr/geometry/dynamic_obstacles/sob_builder.hpp"
#include "vkr/config.hpp"
#include <thread>
#include <random>
#include <fstream>

namespace vkr 
{
namespace geometry 
{
namespace dynamic_obstacles 
{

class SweptOBBBVHTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.max_faces_per_leaf = 4;
    config_.bvh_max_leaf_size = 4;
    config_.bvh_sah_traversal_cost = 1.0f;
    config_.bvh_sah_intersection_cost = 1.5f;
    config_.bvh_use_temporal_splits = true;
    
    builder_ = std::make_unique<SweptOBBBVHBuilder>(config_);
    base_time_ = std::chrono::steady_clock::now();
  }
  
  std::unique_ptr<SweptOBBBVH> createSimpleBVH() 
  {
    builder_->clear();
    
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
      base_time_,
      base_time_ + std::chrono::seconds(10)
    );
    
    return builder_->build();
  }
  
  GeometryConfig config_;
  std::unique_ptr<SweptOBBBVHBuilder> builder_;
  std::chrono::steady_clock::time_point base_time_;
};

TEST_F(SweptOBBBVHTest, EmptyBVH) 
{
  auto bvh = builder_->build();
  
  auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), base_time_);
  EXPECT_FALSE(result.hit);
  
  auto stats = bvh->getStats();
  EXPECT_EQ(stats.num_objects, 0);
  EXPECT_EQ(stats.num_nodes, 0);
}

TEST_F(SweptOBBBVHTest, QueryPoint_Hit) 
{
  auto bvh = createSimpleBVH();
  
  auto result = bvh->queryPoint(Eigen::Vector3f(0.5f, 0.5f, 0.5f), base_time_);
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_id, 1);
  EXPECT_EQ(result.object_type, collision::HitResult::ObjectType::DYNAMIC_OBSTACLE);
  
  EXPECT_TRUE(result.hit_point.allFinite());
  EXPECT_TRUE(result.hit_normal.allFinite());
  EXPECT_NEAR(result.hit_normal.norm(), 1.0f, 1e-3f);
}

TEST_F(SweptOBBBVHTest, QueryPoint_Miss) 
{
  auto bvh = createSimpleBVH();
  
  auto result = bvh->queryPoint(Eigen::Vector3f(5, 0, 0), base_time_);
  EXPECT_FALSE(result.hit);
}

TEST_F(SweptOBBBVHTest, QueryPoint_TimeRange) 
{
  auto bvh = createSimpleBVH();
  
  auto result = bvh->queryPoint(
    Eigen::Vector3f(0, 0, 0), 
    base_time_ - std::chrono::seconds(1)
  );
  EXPECT_FALSE(result.hit);
  
  result = bvh->queryPoint(
    Eigen::Vector3f(0, 0, 0),
    base_time_ + std::chrono::seconds(20)
  );
  EXPECT_FALSE(result.hit);
}

TEST_F(SweptOBBBVHTest, QueryOBB_Hit) 
{
  auto bvh = createSimpleBVH();
  
  OBB query;
  query.center = Eigen::Vector3f(0.5f, 0, 0);
  query.half_extents = Eigen::Vector3f(0.6f, 0.6f, 0.6f);
  query.orientation = Eigen::Matrix3f::Identity();
  
  auto result = bvh->queryOBB(query, base_time_, base_time_);
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_id, 1);
}

TEST_F(SweptOBBBVHTest, QueryOBB_TimeInterval) 
{
  builder_->clear();
  
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {Eigen::Vector3f(-5, 0, 0), Eigen::Vector3f(10, 0, 0)},
    {Eigen::Vector3f(0, 0, 0)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  OBB query;
  query.center = Eigen::Vector3f(0, 0, 0);
  query.half_extents = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
  query.orientation = Eigen::Matrix3f::Identity();
  
  auto result = bvh->queryOBB(
    query,
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  EXPECT_TRUE(result.hit);
}

TEST_F(SweptOBBBVHTest, QueryPath_Simple) 
{
  auto bvh = createSimpleBVH();
  
  std::vector<Eigen::Vector3f> waypoints = 
  {
    Eigen::Vector3f(-3, 0, 0),
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(3, 0, 0)
  };
  
  Eigen::Vector3f half_extents(0.1f, 0.1f, 0.1f);
  
  auto result = bvh->queryPath(waypoints, half_extents, base_time_, 1.0f);
  EXPECT_TRUE(result.hit);
}

TEST_F(SweptOBBBVHTest, QueryPath_Empty) 
{
  auto bvh = createSimpleBVH();
  
  std::vector<Eigen::Vector3f> waypoints;
  Eigen::Vector3f half_extents(0.1f, 0.1f, 0.1f);
  auto result = bvh->queryPath(waypoints, half_extents, base_time_, 1.0f);
  EXPECT_FALSE(result.hit);
  
  waypoints.push_back(Eigen::Vector3f(0, 0, 0));
  result = bvh->queryPath(waypoints, half_extents, base_time_, 1.0f);
  EXPECT_FALSE(result.hit);
}

TEST_F(SweptOBBBVHTest, BatchQueryPoints) 
{
  auto bvh = createSimpleBVH();
  
  const size_t count = 100;
  std::vector<Eigen::Vector3f> points;
  std::vector<collision::HitResult> results(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    float x = (i % 10 - 5) * 0.4f;
    float y = ((i / 10) % 10 - 5) * 0.4f;
    float z = 0.0f;
    points.push_back(Eigen::Vector3f(x, y, z));
  }
  
  bvh->batchQueryPoints(points.data(), count, base_time_, results.data());
  
  size_t hits = 0;
  for (const auto& result : results) 
  {
    if (result.hit) hits++;
  }
  
  EXPECT_GT(hits, 0);
  EXPECT_LT(hits, count);
}

TEST_F(SweptOBBBVHTest, GetActiveOBBs) 
{
  builder_->clear();
  
  for (int i = 0; i < 5; ++i) 
  {
    OBB obb;
    obb.center = Eigen::Vector3f(i, 0, 0);
    obb.half_extents = Eigen::Vector3f(0.4f, 0.4f, 0.4f);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {Eigen::Vector3f(i, 0, 0)},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      base_time_ + std::chrono::seconds(i),
      base_time_ + std::chrono::seconds(i + 5)
    );
  }
  
  auto bvh = builder_->build();
  
  auto active = bvh->getActiveOBBs(base_time_ + std::chrono::seconds(3));
  EXPECT_EQ(active.size(), 4);
}

TEST_F(SweptOBBBVHTest, GetDistance) 
{
  auto bvh = createSimpleBVH();
  
  float dist = bvh->getDistance(Eigen::Vector3f(3, 0, 0), base_time_);
  EXPECT_NEAR(dist, 2.0f, 1e-2f);
  
  dist = bvh->getDistance(Eigen::Vector3f(0, 0, 0), base_time_);
  EXPECT_LE(dist, 0.0f);
}

TEST_F(SweptOBBBVHTest, LargeScenario) 
{
  builder_->clear();
  
  const int grid_size = 10;
  for (int x = 0; x < grid_size; ++x) 
  {
    for (int y = 0; y < grid_size; ++y) 
    {
      float start_x = x * 3.0f;
      float start_y = y * 3.0f;
      
      OBB obb;
      obb.center = Eigen::Vector3f(start_x, start_y, 0);
      obb.half_extents = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
      obb.orientation = Eigen::Matrix3f::Identity();
      
      builder_->addWithTimeMapping(
        obb,
        x * grid_size + y,
        {
          Eigen::Vector3f(start_x, start_y, 0),
          Eigen::Vector3f(start_x + (x - 5) * 0.1f, start_y + (y - 5) * 0.1f, 0)
        },
        {Eigen::Vector3f(0, 0, 0)},
        {},
        base_time_,
        base_time_ + std::chrono::seconds(20)
      );
    }
  }
  
  auto bvh = builder_->build();
  auto stats = bvh->getStats();
  
  EXPECT_EQ(stats.num_objects, grid_size * grid_size);
  EXPECT_GT(stats.tree_depth, 3);
  EXPECT_GT(stats.num_nodes, 1);
  EXPECT_GT(stats.num_leaves, 1);
  EXPECT_GE(stats.num_nodes, stats.num_leaves);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 1000; ++i) 
  {
    Eigen::Vector3f query_point(
      (i % 30) * 1.0f,
      ((i / 30) % 30) * 1.0f,
      0.0f
    );
    bvh->queryPoint(query_point, base_time_ + std::chrono::seconds(10));
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  EXPECT_LT(duration / 1000.0, 1000.0);
}

TEST_F(SweptOBBBVHTest, RotatingOBBs) 
{
  builder_->clear();
  
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(2, 0.5f, 0.5f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 2 * M_PI)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  auto result1 = bvh->queryPoint(Eigen::Vector3f(1.5f, 0, 0), base_time_);
  EXPECT_TRUE(result1.hit);
  
  auto quarter_time = base_time_ + std::chrono::milliseconds(2500);
  auto result2 = bvh->queryPoint(Eigen::Vector3f(1.5f, 0, 0), quarter_time);
  EXPECT_FALSE(result2.hit);
  
  auto result3 = bvh->queryPoint(Eigen::Vector3f(0, 1.5f, 0), quarter_time);
  EXPECT_TRUE(result3.hit);
}

TEST_F(SweptOBBBVHTest, TemporalSplits) 
{
  builder_->clear();
  
  for (int i = 0; i < 10; ++i) 
  {
    OBB obb;
    obb.center = Eigen::Vector3f(0, 0, 0);
    obb.half_extents = Eigen::Vector3f(1, 1, 1);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {Eigen::Vector3f(0, 0, 0)},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      base_time_ + std::chrono::seconds(i * 2),
      base_time_ + std::chrono::seconds(i * 2 + 1)
    );
  }
  
  auto bvh = builder_->build();
  
  for (int i = 0; i < 10; ++i) 
  {
    auto query_time = base_time_ + std::chrono::milliseconds(i * 2000 + 500);
    auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), query_time);
    
    if (result.hit) 
    {
      EXPECT_EQ(result.object_id, i);
    }
  }
}

TEST_F(SweptOBBBVHTest, RepeatingTrajectories) 
{
  builder_->clear();
  
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {
      Eigen::Vector3f(0, 0, 0),
      Eigen::Vector3f(1, 0, 0),
      Eigen::Vector3f(0, 1, 0),
      Eigen::Vector3f(-1, 0, 0)
    },
    {Eigen::Vector3f(0, 0, 0)},
    {0.0f, 0.1f},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  auto future = base_time_ + std::chrono::seconds(100);
  auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), future);
  EXPECT_TRUE(result.object_type == collision::HitResult::ObjectType::DYNAMIC_OBSTACLE || 
              result.object_type == collision::HitResult::ObjectType::NONE);
}

TEST_F(SweptOBBBVHTest, ExportToJSON) 
{
  builder_->clear();
  for (int i = 0; i < 8; ++i) 
  {
    float angle = i * M_PI / 4;
    OBB obb;
    obb.center = Eigen::Vector3f(cos(angle) * 5, sin(angle) * 5, 0);
    obb.half_extents = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {obb.center},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      base_time_,
      base_time_ + std::chrono::seconds(10)
    );
  }
  
  auto bvh = builder_->build();
  
  bvh->exportToJSON("/tmp/test_obb_bvh.json");
  
  std::ifstream file("/tmp/test_obb_bvh.json");
  EXPECT_TRUE(file.good());
  
  std::remove("/tmp/test_obb_bvh.json");
}

TEST_F(SweptOBBBVHTest, ConcurrentQueries) 
{
  builder_->clear();
  for (int i = 0; i < 50; ++i) 
  {
    OBB obb;
    obb.center = Eigen::Vector3f(i % 10, i / 10, 0);
    obb.half_extents = Eigen::Vector3f(0.3f, 0.3f, 0.3f);
    obb.orientation = Eigen::Matrix3f::Identity();
    
    builder_->addWithTimeMapping(
      obb,
      i,
      {obb.center},
      {Eigen::Vector3f(0, 0, 0)},
      {},
      base_time_,
      base_time_ + std::chrono::seconds(30)
    );
  }
  
  auto bvh = builder_->build();
  
  const int num_threads = 4;
  const int queries_per_thread = 1000;
  std::vector<std::thread> threads;
  std::atomic<int> total_hits{0};
  
  for (int t = 0; t < num_threads; ++t) 
  {
    threads.emplace_back([&, t]() 
    {
      std::mt19937 rng(t);
      std::uniform_real_distribution<float> dist(-5, 15);
      
      int hits = 0;
      for (int i = 0; i < queries_per_thread; ++i) 
      {
        Eigen::Vector3f point(dist(rng), dist(rng), 0.0f);
        auto result = bvh->queryPoint(point, base_time_ + std::chrono::seconds(15));
        if (result.hit) hits++;
      }
      
      total_hits += hits;
    });
  }
  
  for (auto& thread : threads) 
  {
    thread.join();
  }
  
  EXPECT_GT(total_hits.load(), 0);
  EXPECT_LT(total_hits.load(), num_threads * queries_per_thread);
}

TEST_F(SweptOBBBVHTest, EdgeCases_ZeroSizeOBB) 
{
  builder_->clear();
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(0, 0, 0);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), base_time_);
  EXPECT_TRUE(result.hit);
  
  result = bvh->queryPoint(Eigen::Vector3f(0.1f, 0, 0), base_time_);
  EXPECT_FALSE(result.hit);
}

TEST_F(SweptOBBBVHTest, ComplexTrajectories) 
{
  builder_->clear();
  
  std::vector<Eigen::Vector3f> pos_coeffs = 
  {
    Eigen::Vector3f(0.0f, 0.0f, 0.0f),
    Eigen::Vector3f(5.0f * M_PI, 0.0f, 0.0f),
    Eigen::Vector3f(0.0f, 0.0f, 0.0f),
    Eigen::Vector3f(-std::pow(5.0 * M_PI, 3) / 6.0f, 0.0f, 0.0f)
  };
  
  std::vector<Eigen::Vector3f> rot_coeffs;
  for (int i = 0; i <= 5; ++i) 
  {
    rot_coeffs.push_back(Eigen::Vector3f(
      0,
      0,
      std::pow(-1, i) * 0.5f / (i + 1)
    ));
  }
  
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 0.5f, 0.25f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    pos_coeffs,
    rot_coeffs,
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  const int num_samples = 100;
  int hits = 0;
  
  for (int i = 0; i < num_samples; ++i) 
  {
    auto t = base_time_ + std::chrono::milliseconds(i * 100);
    auto result = bvh->queryPoint(Eigen::Vector3f(0, 0, 0), t);
    if (result.hit) hits++;
  }
  
  EXPECT_GT(hits, 0);
  EXPECT_LT(hits, num_samples);
}

TEST_F(SweptOBBBVHTest, OBBvsOBB_ComplexRotations) 
{
  builder_->clear();
  
  OBB obb1;
  obb1.center = Eigen::Vector3f(0, 0, 0);
  obb1.half_extents = Eigen::Vector3f(2, 0.2f, 0.2f);
  obb1.orientation = Eigen::Matrix3f::Identity();
  
  OBB obb2;
  obb2.center = Eigen::Vector3f(0, 0, 0);
  obb2.half_extents = Eigen::Vector3f(2, 0.2f, 0.2f);
  obb2.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    obb1,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, M_PI)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  builder_->addWithTimeMapping(
    obb2,
    2,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, M_PI/2), Eigen::Vector3f(0, 0, -M_PI/2)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  OBB query;
  query.center = Eigen::Vector3f(1.5f, 1.5f, 0);
  query.half_extents = Eigen::Vector3f(0.1f, 0.1f, 0.1f);
  query.orientation = Eigen::Matrix3f::Identity();
  
  auto result = bvh->queryOBB(query, base_time_, base_time_);
  EXPECT_FALSE(result.hit);
  
  auto mid_time = base_time_ + std::chrono::seconds(5);
  query.center = Eigen::Vector3f(0, 0, 0);
  result = bvh->queryOBB(query, mid_time, mid_time);
  EXPECT_TRUE(result.hit);
}

TEST_F(SweptOBBBVHTest, NonAxisAlignedOBBs) 
{
  builder_->clear();
  
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(2, 1, 0.5f);
  
  float angle = M_PI / 4;
  obb.orientation = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  
  builder_->addWithTimeMapping(
    obb,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  float sqrt2 = std::sqrt(2.0f);
  auto result1 = bvh->queryPoint(Eigen::Vector3f(sqrt2, sqrt2, 0) * 0.9f, base_time_);
  EXPECT_TRUE(result1.hit);
  
  auto result2 = bvh->queryPoint(Eigen::Vector3f(-sqrt2, -sqrt2, 0) * 0.9f, base_time_);
  EXPECT_TRUE(result2.hit);
  
  auto result3 = bvh->queryPoint(Eigen::Vector3f(2, 0, 0), base_time_);
  EXPECT_FALSE(result3.hit);
  
  auto result4 = bvh->queryPoint(Eigen::Vector3f(0, 2, 0), base_time_);
  EXPECT_FALSE(result4.hit);
}

TEST_F(SweptOBBBVHTest, PathQueryWithRotatingObject) 
{
  builder_->clear();
  
  OBB barrier;
  barrier.center = Eigen::Vector3f(0, 0, 0);
  barrier.half_extents = Eigen::Vector3f(3, 0.2f, 1);
  barrier.orientation = Eigen::Matrix3f::Identity();
  
  builder_->addWithTimeMapping(
    barrier,
    1,
    {Eigen::Vector3f(0, 0, 0)},
    {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, M_PI/2)},
    {},
    base_time_,
    base_time_ + std::chrono::seconds(10)
  );
  
  auto bvh = builder_->build();
  
  std::vector<Eigen::Vector3f> waypoints = 
  {
    Eigen::Vector3f(-5, 0, 0),
    Eigen::Vector3f(5, 0, 0)
  };
  
  Eigen::Vector3f half_extents(0.5f, 0.5f, 0.5f);
  
  auto result1 = bvh->queryPath(waypoints, half_extents, base_time_, 1.0f);
  EXPECT_TRUE(result1.hit);
  
  auto result2 = bvh->queryPath(waypoints, half_extents, base_time_ + std::chrono::seconds(10), 1.0f);
  EXPECT_FALSE(result2.hit);
}

}
}
}