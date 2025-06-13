#include <gtest/gtest.h>
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/config.hpp"
#include <thread>
#include <random>
#include <chrono>
#include <fstream>

namespace vkr 
{
namespace geometry 
{
namespace corridors 
{

class SpatialCorridorMapTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.grid_cell_size = 4.0f;
    config_.hash_table_size = 1024;
    config_.corridor_radius = 2.5f;
    config_.corridor_coverage_fraction = 0.8f;
    
    map_ = std::make_unique<SpatialCorridorMap>(config_);
  }
  
  std::vector<Eigen::Vector3f> createPath(const Eigen::Vector3f& start, 
                                          const Eigen::Vector3f& end, 
                                          int segments) 
  {
    std::vector<Eigen::Vector3f> path;
    for (int i = 0; i <= segments; ++i) 
    {
      float t = static_cast<float>(i) / segments;
      path.push_back(start * (1 - t) + end * t);
    }
    return path;
  }
  
  std::vector<Eigen::Vector3f> createCircularPath(const Eigen::Vector3f& center,
                                                  float radius,
                                                  int segments) 
  {
    std::vector<Eigen::Vector3f> path;
    for (int i = 0; i <= segments; ++i) 
    {
      float angle = 2.0f * M_PI * i / segments;
      path.push_back(center + Eigen::Vector3f(radius * cos(angle), 
                                              radius * sin(angle), 0));
    }
    return path;
  }
  
  GeometryConfig config_;
  std::unique_ptr<SpatialCorridorMap> map_;
};

TEST_F(SpatialCorridorMapTest, Construction) 
{
  EXPECT_NE(map_, nullptr);
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, 0);
  EXPECT_EQ(stats.total_segments, 0);
  EXPECT_EQ(stats.active_segments, 0);
  EXPECT_EQ(stats.freed_segments, 0);
}

TEST_F(SpatialCorridorMapTest, CreateCorridor_ValidPath) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 10);
  
  CorridorId id = map_->createCorridor(1, path, 2.0f, 0);
  
  EXPECT_NE(id, INVALID_ID);
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, 1);
  EXPECT_EQ(stats.total_segments, 8);
  EXPECT_EQ(stats.active_segments, 8);
}

TEST_F(SpatialCorridorMapTest, CreateCorridor_CoverageFraction) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(100, 0, 0), 100);
  
  CorridorId id = map_->createCorridor(1, path, 2.0f, 20);
  
  EXPECT_NE(id, INVALID_ID);
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_segments, 64);
  
  Corridor info;
  EXPECT_TRUE(map_->getCorridorInfo(id, info));
  EXPECT_EQ(info.current_waypoint, 20);
  EXPECT_EQ(info.total_waypoints, 101);
  EXPECT_FLOAT_EQ(info.coverage_fraction, 0.8f);
}

TEST_F(SpatialCorridorMapTest, CreateCorridor_InvalidPath) 
{
  std::vector<Eigen::Vector3f> empty_path;
  CorridorId id1 = map_->createCorridor(1, empty_path, 2.0f);
  EXPECT_EQ(id1, INVALID_ID);
  
  std::vector<Eigen::Vector3f> single_point = {Eigen::Vector3f(0, 0, 0)};
  CorridorId id2 = map_->createCorridor(1, single_point, 2.0f);
  EXPECT_EQ(id2, INVALID_ID);
  
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  CorridorId id3 = map_->createCorridor(1, path, 2.0f, 5);
  EXPECT_EQ(id3, INVALID_ID);
  
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, 0);
}

TEST_F(SpatialCorridorMapTest, UpdateCorridorCoverage) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(20, 0, 0), 20);
  
  CorridorId id = map_->createCorridor(1, path, 2.0f, 0);
  EXPECT_NE(id, INVALID_ID);
  
  auto stats = map_->getStats();
  EXPECT_EQ(stats.active_segments, 16);
  
  map_->updateCorridorCoverage(id, path, 5);
  
  stats = map_->getStats();
  EXPECT_LE(stats.active_segments, 12);
  EXPECT_GT(stats.freed_segments, 0);
  
  map_->updateCorridorCoverage(id, path, 20);
  
  stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, 0);
}

TEST_F(SpatialCorridorMapTest, RemoveCorridor) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  CorridorId id = map_->createCorridor(1, path, 2.0f);
  
  auto stats = map_->getStats();
  size_t initial_segments = stats.active_segments;
  
  map_->removeCorridor(id);
  
  stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, 0);
  EXPECT_EQ(stats.active_segments, 0);
  EXPECT_EQ(stats.total_segments, initial_segments);
}

TEST_F(SpatialCorridorMapTest, IsInsideOwnCorridor) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 2);
  map_->createCorridor(1, path, 2.0f);
  
  EXPECT_TRUE(map_->isInsideOwnCorridor(Eigen::Vector3f(5, 0, 0), 1));
  EXPECT_TRUE(map_->isInsideOwnCorridor(Eigen::Vector3f(5, 1.5f, 0), 1));
  
  EXPECT_FALSE(map_->isInsideOwnCorridor(Eigen::Vector3f(5, 3, 0), 1));
  EXPECT_FALSE(map_->isInsideOwnCorridor(Eigen::Vector3f(-3, 0, 0), 1)); 
  
  EXPECT_FALSE(map_->isInsideOwnCorridor(Eigen::Vector3f(5, 0, 0), 2));
}

TEST_F(SpatialCorridorMapTest, IsInsideOtherCorridor) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 2);
  map_->createCorridor(1, path, 2.0f);
  
  EXPECT_TRUE(map_->isInsideOtherCorridor(Eigen::Vector3f(5, 0, 0), 2));
  EXPECT_FALSE(map_->isInsideOtherCorridor(Eigen::Vector3f(5, 3, 0), 2));
  
  EXPECT_FALSE(map_->isInsideOtherCorridor(Eigen::Vector3f(5, 0, 0), 1));
}

TEST_F(SpatialCorridorMapTest, GetDistanceToBoundary) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 2);
  map_->createCorridor(1, path, 2.0f);
  
  float dist = map_->getDistanceToBoundary(Eigen::Vector3f(5, 0, 0), 1);
  EXPECT_NEAR(dist, 2.0f, 0.1f);
  
  dist = map_->getDistanceToBoundary(Eigen::Vector3f(5, 1.5f, 0), 1);
  EXPECT_NEAR(dist, 0.5f, 0.1f);
  
  dist = map_->getDistanceToBoundary(Eigen::Vector3f(5, 3, 0), 1);
  EXPECT_NEAR(dist, 1.0f, 0.1f);
}

TEST_F(SpatialCorridorMapTest, IntersectCapsule) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 2);
  map_->createCorridor(1, path, 2.0f);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(5, 0, 0);
  capsule.p1 = Eigen::Vector3f(5, 5, 0);
  capsule.radius = 1.0f;
  
  EXPECT_TRUE(map_->intersectCapsule(capsule));
  EXPECT_FALSE(map_->intersectCapsule(capsule, 1));
  
  capsule.p0 = Eigen::Vector3f(5, 5, 0);
  capsule.p1 = Eigen::Vector3f(5, 10, 0);
  EXPECT_FALSE(map_->intersectCapsule(capsule));
}

TEST_F(SpatialCorridorMapTest, IntersectSegment) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 2);
  map_->createCorridor(1, path, 2.0f);
  
  LineSegment segment;
  segment.start = Eigen::Vector3f(5, -5, 0);
  segment.end = Eigen::Vector3f(5, 5, 0);
  
  EXPECT_TRUE(map_->intersectSegment(segment));
  
  segment.start = Eigen::Vector3f(5, 3, 0);
  segment.end = Eigen::Vector3f(5, 5, 0);
  EXPECT_FALSE(map_->intersectSegment(segment));
}

TEST_F(SpatialCorridorMapTest, GetActiveCorridors) 
{
  UAVId uav1 = 1;
  UAVId uav2 = 2;
  
  auto path1 = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  auto path2 = createPath(Eigen::Vector3f(0, 5, 0), Eigen::Vector3f(10, 5, 0), 5);
  
  CorridorId id1 = map_->createCorridor(uav1, path1, 2.0f);
  CorridorId id2 = map_->createCorridor(uav1, path2, 2.0f);
  CorridorId id3 = map_->createCorridor(uav2, path1, 2.0f);
  
  auto corridors_uav1 = map_->getActiveCorridors(uav1);
  EXPECT_EQ(corridors_uav1.size(), 2);
  EXPECT_TRUE(std::find(corridors_uav1.begin(), corridors_uav1.end(), id1) != corridors_uav1.end());
  EXPECT_TRUE(std::find(corridors_uav1.begin(), corridors_uav1.end(), id2) != corridors_uav1.end());
  
  auto corridors_uav2 = map_->getActiveCorridors(uav2);
  EXPECT_EQ(corridors_uav2.size(), 1);
  EXPECT_EQ(corridors_uav2[0], id3);
}

TEST_F(SpatialCorridorMapTest, FreeSegmentEvents) 
{
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  CorridorId id = map_->createCorridor(1, path, 2.0f);
  
  Corridor info;
  EXPECT_TRUE(map_->getCorridorInfo(id, info));
  EXPECT_GT(info.active_segments.size(), 0);
  
  SegmentId first_segment = info.active_segments[0];
  
  map_->freeSegment(id, first_segment);
  
  CorridorFreeEvent event;
  EXPECT_TRUE(map_->getNextFreeEvent(event));
  EXPECT_EQ(event.corridor_id, id);
  EXPECT_EQ(event.segment_id, first_segment);
  
  EXPECT_FALSE(map_->getNextFreeEvent(event));
}

TEST_F(SpatialCorridorMapTest, MultipleCorridorsOverlap) 
{
  auto path1 = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  auto path2 = createPath(Eigen::Vector3f(5, -5, 0), Eigen::Vector3f(5, 5, 0), 5);
  
  map_->createCorridor(1, path1, 2.0f);
  map_->createCorridor(2, path2, 2.0f);
  
  Eigen::Vector3f intersection(5, 0, 0);
  EXPECT_TRUE(map_->isInsideOwnCorridor(intersection, 1));
  EXPECT_TRUE(map_->isInsideOtherCorridor(intersection, 1));
  EXPECT_TRUE(map_->isInsideOwnCorridor(intersection, 2));
  EXPECT_TRUE(map_->isInsideOtherCorridor(intersection, 2));
}

TEST_F(SpatialCorridorMapTest, LargeScale) 
{
  const int num_uavs = 10;
  const int segments_per_path = 50;
  
  for (int i = 0; i < num_uavs; ++i) 
  {
    float y_offset = i * 10.0f;
    auto path = createPath(
      Eigen::Vector3f(0, y_offset, 0),
      Eigen::Vector3f(100, y_offset, 0),
      segments_per_path
    );
    
    CorridorId id = map_->createCorridor(i, path, 2.0f);
    EXPECT_NE(id, INVALID_ID);
  }
  
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, num_uavs);
  size_t expected_segments = num_uavs * static_cast<size_t>(segments_per_path * 0.8);
  EXPECT_NEAR(stats.total_segments, expected_segments, num_uavs);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < 1000; ++i) 
  {
    Eigen::Vector3f pos(50, i * 0.1f, 0);
    map_->isInsideOwnCorridor(pos, 0);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  EXPECT_LT(duration / 1000.0, 1000.0);
}

TEST_F(SpatialCorridorMapTest, CircularPath) 
{
  auto path = createCircularPath(Eigen::Vector3f(0, 0, 0), 10.0f, 20);
  
  CorridorId id = map_->createCorridor(1, path, 2.0f);
  EXPECT_NE(id, INVALID_ID);
  
  EXPECT_TRUE(map_->isInsideOwnCorridor(Eigen::Vector3f(10, 0, 0), 1));
  EXPECT_TRUE(map_->isInsideOwnCorridor(Eigen::Vector3f(0, 10, 0), 1));
  EXPECT_TRUE(map_->isInsideOwnCorridor(Eigen::Vector3f(-10, 0, 0), 1));
  
  EXPECT_FALSE(map_->isInsideOwnCorridor(Eigen::Vector3f(0, 0, 0), 1));
  
  EXPECT_FALSE(map_->isInsideOwnCorridor(Eigen::Vector3f(15, 0, 0), 1));
}

TEST_F(SpatialCorridorMapTest, ConcurrentAccess) 
{
  const int num_threads = 4;
  const int ops_per_thread = 100;
  
  std::vector<std::thread> threads;
  std::atomic<int> corridor_count{0};
  
  for (int t = 0; t < num_threads; ++t) 
  {
    threads.emplace_back([&, t]() 
    {
      for (int i = 0; i < ops_per_thread; ++i) 
      {
        float offset = t * 20.0f + i * 0.1f;
        auto path = createPath(
          Eigen::Vector3f(offset, 0, 0),
          Eigen::Vector3f(offset + 10, 0, 0),
          5
        );
        
        CorridorId id = map_->createCorridor(t, path, 1.0f);
        if (id != INVALID_ID) 
        {
          corridor_count++;
        }
        
        map_->isInsideOwnCorridor(Eigen::Vector3f(offset + 5, 0, 0), t);
      }
    });
  }
  
  for (auto& thread : threads) 
  {
    thread.join();
  }
  
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, corridor_count.load());
}

TEST_F(SpatialCorridorMapTest, EdgeCases) 
{
  auto path1 = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0.1f, 0, 0), 2);
  CorridorId id1 = map_->createCorridor(1, path1, 0.05f);
  EXPECT_NE(id1, INVALID_ID);
  
  auto path2 = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1000, 0, 0), 100);
  CorridorId id2 = map_->createCorridor(2, path2, 50.0f);
  EXPECT_NE(id2, INVALID_ID);
  
  std::vector<Eigen::Vector3f> degen_path = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(1, 0, 0)
  };
  CorridorId id3 = map_->createCorridor(3, degen_path, 1.0f);
  EXPECT_NE(id3, INVALID_ID);
}

TEST_F(SpatialCorridorMapTest, ExportToJSON) 
{
  auto path1 = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 5);
  auto path2 = createCircularPath(Eigen::Vector3f(20, 0, 0), 5.0f, 10);
  
  map_->createCorridor(1, path1, 2.0f);
  map_->createCorridor(2, path2, 1.5f);
  
  std::string filename = "/tmp/test_corridor_map.json";
  map_->exportToJSON(filename);
  
  std::ifstream file(filename);
  EXPECT_TRUE(file.good());
  
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  EXPECT_FALSE(content.empty());
  
  std::remove(filename.c_str());
}

TEST_F(SpatialCorridorMapTest, MemoryManagement) 
{
  const int iterations = 100;
  
  for (int i = 0; i < iterations; ++i) 
  {
    auto path = createPath(
      Eigen::Vector3f(i, 0, 0),
      Eigen::Vector3f(i + 10, 0, 0),
      10
    );
    
    CorridorId id = map_->createCorridor(i % 10, path, 1.0f);
    
    if (i % 2 == 1) 
    {
      map_->removeCorridor(id);
    }
  }
  
  auto stats = map_->getStats();
  EXPECT_EQ(stats.total_corridors, iterations / 2);
  EXPECT_GT(stats.freed_segments, 0);
}

TEST_F(SpatialCorridorMapTest, CoverageFractionBoundary) 
{
  config_.corridor_coverage_fraction = 1.0f;
  auto map_full = std::make_unique<SpatialCorridorMap>(config_);
  
  config_.corridor_coverage_fraction = 0.5f;
  auto map_half = std::make_unique<SpatialCorridorMap>(config_);
  
  auto path = createPath(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(20, 0, 0), 20);
  
  map_full->createCorridor(1, path, 2.0f);
  map_half->createCorridor(1, path, 2.0f);
  
  auto stats_full = map_full->getStats();
  auto stats_half = map_half->getStats();
  
  EXPECT_EQ(stats_full.active_segments, 20);
  EXPECT_EQ(stats_half.active_segments, 10);
}

}
}
}