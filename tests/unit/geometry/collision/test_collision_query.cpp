#include <gtest/gtest.h>
#include "vkr/geometry/collision/collision_query.hpp"
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/terrain/qwg_builder.hpp"
#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/geometry/static_obstacles/bst_builder.hpp"
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/config.hpp"
#include <gdal_priv.h>
#include <filesystem>
#include <thread>
#include <random>
#include <fstream>
#include <spdlog/spdlog.h>

namespace vkr
{
namespace geometry
{
namespace collision
{

namespace fs = std::filesystem;

class CollisionQueryTest : public ::testing::Test 
{
protected:
  static void SetUpTestSuite() 
  {
    test_dir_ = fs::temp_directory_path() / "vkr_collision_test";
    fs::create_directories(test_dir_);
    
    GDALAllRegister();
    
    createTestTerrain();
  }
  
  static void TearDownTestSuite() 
  {
    if (fs::exists(test_dir_)) 
    {
      fs::remove_all(test_dir_);
    }
  }
  
  void SetUp() override 
  {
    config_.grid_cell_size = 8.0f;
    config_.hash_table_size = 65536;
    config_.max_faces_per_leaf = 16;
    
    collision_query_ = std::make_unique<CollisionQuery>(config_);
  }
  
  static void createTestTerrain() 
  {
    constexpr size_t size = 1024;
    std::string tif_file = (test_dir_ / "test_terrain.tif").string();
    
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* dataset = driver->Create(tif_file.c_str(), size, size, 
                                         1, GDT_Float32, nullptr);
    
    double geotransform[6] = 
    {
      0, 1, 0, size, 0, -1
    };
    dataset->SetGeoTransform(geotransform);
    
    std::vector<float> buffer(size * size);
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0, 2.0f);
    
    for (size_t y = 0; y < size; ++y) 
    {
      for (size_t x = 0; x < size; ++x) 
      {
        float h = 100.0f + 0.05f * x + 0.03f * y;
        
        for (int i = 0; i < 5; ++i) 
        {
          float hx = 200 + i * 150;
          float hy = 200 + i * 100;
          float dx = x - hx;
          float dy = y - hy;
          h += 30.0f * std::exp(-(dx*dx + dy*dy) / 5000.0f);
        }
        
        if (x > 400 && x < 450 && y > 100 && y < 900) 
        {
          h -= 40.0f;
        }
        
        h += noise(rng);
        
        buffer[y * size + x] = h;
      }
    }
    
    GDALRasterBand* band = dataset->GetRasterBand(1);
    band->RasterIO(GF_Write, 0, 0, size, size, 
                   buffer.data(), size, size, GDT_Float32, 0, 0);
    band->SetNoDataValue(-9999.0);
    
    GDALClose(dataset);
    
    terrain_file_ = (test_dir_ / "test_terrain.qwg").string();
    
    TerrainConfig terrain_config;
    terrain_config.wavelet_levels = 6;
    terrain_config.page_size = 8192;
    terrain_config.max_cache_size = 100;
    
    terrain::QWGBuilder builder(terrain_config);
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(tif_file, terrain_file_, error_message);
    if (!success) 
    {
      throw std::runtime_error("Failed to create test terrain: " + error_message);
    }
  }
  
  std::shared_ptr<static_obstacles::BoundingSphereTree> createStaticObstacles() 
  {
    static_obstacles::BSTBuilder builder(config_);
    
    float cx = 50, cy = 50, cz = 100;
    float s = 10;
    
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz-s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz-s),
      Eigen::Vector3f(cx-s, cy+s, cz-s),
      1
    );
    
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz+s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      Eigen::Vector3f(cx+s, cy-s, cz+s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz+s),
      Eigen::Vector3f(cx-s, cy+s, cz+s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      1
    );
    
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy-s, cz+s),
      Eigen::Vector3f(cx+s, cy-s, cz-s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx-s, cy-s, cz+s),
      Eigen::Vector3f(cx+s, cy-s, cz+s),
      1
    );
    
    builder.addFace(
      Eigen::Vector3f(cx-s, cy+s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx-s, cy+s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      Eigen::Vector3f(cx-s, cy+s, cz+s),
      1
    );
    
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx-s, cy+s, cz-s),
      Eigen::Vector3f(cx-s, cy+s, cz+s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx-s, cy-s, cz-s),
      Eigen::Vector3f(cx-s, cy+s, cz+s),
      Eigen::Vector3f(cx-s, cy-s, cz+s),
      1
    );
    
    builder.addFace(
      Eigen::Vector3f(cx+s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      Eigen::Vector3f(cx+s, cy+s, cz-s),
      1
    );
    builder.addFace(
      Eigen::Vector3f(cx+s, cy-s, cz-s),
      Eigen::Vector3f(cx+s, cy-s, cz+s),
      Eigen::Vector3f(cx+s, cy+s, cz+s),
      1
    );
    
    const int segments = 20;
    for (int lat = 0; lat < segments; ++lat) 
    {
      for (int lon = 0; lon < segments; ++lon) 
      {
        float theta1 = lat * M_PI / segments;
        float theta2 = (lat + 1) * M_PI / segments;
        float phi1 = lon * 2 * M_PI / segments;
        float phi2 = (lon + 1) * 2 * M_PI / segments;
        
        auto spherePoint = [](float theta, float phi, float r, const Eigen::Vector3f& center) -> Eigen::Vector3f
        {
          Eigen::Vector3f offset;
          offset.x() = r * std::sin(theta) * std::cos(phi);
          offset.y() = r * std::sin(theta) * std::sin(phi);
          offset.z() = r * std::cos(theta);
          return center + offset;
        };
        
        Eigen::Vector3f center(150, 150, 120);
        float radius = 15;
        
        Eigen::Vector3f p1 = spherePoint(theta1, phi1, radius, center);
        Eigen::Vector3f p2 = spherePoint(theta2, phi1, radius, center);
        Eigen::Vector3f p3 = spherePoint(theta2, phi2, radius, center);
        Eigen::Vector3f p4 = spherePoint(theta1, phi2, radius, center);
        
        builder.addFace(p1, p2, p3, 2);
        builder.addFace(p1, p3, p4, 2);
      }
    }
    
    return builder.build();
  }
  
  std::shared_ptr<corridors::SpatialCorridorMap> createCorridors() 
  {
    auto corridors = std::make_shared<corridors::SpatialCorridorMap>(config_);
    
    std::vector<Eigen::Vector3f> path1 = 
    {
      Eigen::Vector3f(0, 0, 150),
      Eigen::Vector3f(100, 0, 150),
      Eigen::Vector3f(100, 100, 150),
      Eigen::Vector3f(200, 100, 150)
    };
    corridors->createCorridor(1, path1, 10.0f);
    
    std::vector<Eigen::Vector3f> path2 = 
    {
      Eigen::Vector3f(50, 50, 160),
      Eigen::Vector3f(50, 150, 160),
      Eigen::Vector3f(150, 150, 160)
    };
    corridors->createCorridor(2, path2, 8.0f);
    
    return corridors;
  }
  
  GeometryConfig config_;
  std::unique_ptr<CollisionQuery> collision_query_;
  static fs::path test_dir_;
  static std::string terrain_file_;
};

fs::path CollisionQueryTest::test_dir_;
std::string CollisionQueryTest::terrain_file_;

TEST_F(CollisionQueryTest, Initialization) 
{
  EXPECT_NO_THROW(collision_query_->exportToJSON((test_dir_ / "init.json").string()));
  
  Capsule test_capsule;
  test_capsule.p0 = Eigen::Vector3f(50, 50, 100);
  test_capsule.p1 = Eigen::Vector3f(50, 50, 110);
  test_capsule.radius = 1.0f;
  
  auto result = collision_query_->queryCapsule(test_capsule, std::chrono::steady_clock::now());
  EXPECT_FALSE(result.hit);
}

TEST_F(CollisionQueryTest, SetComponents) 
{
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  auto corridors = createCorridors();
  collision_query_->setCorridors(corridors);
  
  collision_query_->exportToJSON((test_dir_ / "all_components.json").string());
}

TEST_F(CollisionQueryTest, TerrainCollision) 
{
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  float actual_height = terrain->getHeight(128, 128);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(128, 128, actual_height - 10);
  capsule.p1 = Eigen::Vector3f(128, 128, actual_height + 10);
  capsule.radius = 5.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::TERRAIN);
  EXPECT_NEAR(result.hit_point.z(), actual_height, 20.0f);
  EXPECT_GT(result.hit_normal.z(), 0.5f);
}

TEST_F(CollisionQueryTest, TerrainOutOfBounds) 
{
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(2000, 2000, 100);
  capsule.p1 = Eigen::Vector3f(2000, 2000, 110);
  capsule.radius = 5.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  EXPECT_FALSE(result.hit);
}

TEST_F(CollisionQueryTest, StaticObstacleCollision) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(50, 50, 95);
  capsule.p1 = Eigen::Vector3f(50, 50, 105);
  capsule.radius = 1.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::STATIC_OBSTACLE);
  EXPECT_EQ(result.object_id, 1);
}

TEST_F(CollisionQueryTest, StaticObstacleMiss) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(250, 250, 150);
  capsule.p1 = Eigen::Vector3f(250, 250, 160);
  capsule.radius = 1.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  EXPECT_FALSE(result.hit);
}

TEST_F(CollisionQueryTest, CorridorCollision) 
{
  auto corridors = createCorridors();
  collision_query_->setCorridors(corridors);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(100, 50, 145);
  capsule.p1 = Eigen::Vector3f(100, 50, 155);
  capsule.radius = 5.0f;
  
  UAVId uav_id = 3;
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now(), uav_id);
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::CORRIDOR);
}

TEST_F(CollisionQueryTest, OwnCorridorNoCollision) 
{
  auto corridors = createCorridors();
  collision_query_->setCorridors(corridors);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(100, 50, 145);
  capsule.p1 = Eigen::Vector3f(100, 50, 155);
  capsule.radius = 5.0f;
  
  UAVId uav_id = 1;
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now(), uav_id);
  
  EXPECT_FALSE(result.hit);
}

TEST_F(CollisionQueryTest, CollisionPriority) 
{
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  auto corridors = createCorridors();
  collision_query_->setCorridors(corridors);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(50, 50, 80);
  capsule.p1 = Eigen::Vector3f(50, 50, 120);
  capsule.radius = 5.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::TERRAIN);
}

TEST_F(CollisionQueryTest, SegmentQuery) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  LineSegment segment;
  segment.start = Eigen::Vector3f(50, 50, 80);
  segment.end = Eigen::Vector3f(50, 50, 120);
  
  auto result = collision_query_->querySegment(segment, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::STATIC_OBSTACLE);
}

TEST_F(CollisionQueryTest, PathQuery) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  std::vector<Eigen::Vector3f> path = 
  {
    Eigen::Vector3f(0, 50, 100),
    Eigen::Vector3f(50, 50, 100),
    Eigen::Vector3f(100, 50, 100)
  };
  
  auto start_time = std::chrono::steady_clock::now();
  float velocity = 10.0f;
  float radius = 2.0f;
  
  auto result = collision_query_->queryPath(path, radius, start_time, velocity);
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::STATIC_OBSTACLE);
  EXPECT_GT(result.t_min, 0.0f);
  EXPECT_LT(result.t_min, 1.0f);
}

TEST_F(CollisionQueryTest, IsFreeMethod) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  EXPECT_FALSE(collision_query_->isFree(
    Eigen::Vector3f(50, 50, 100), 1.0f, std::chrono::steady_clock::now()
  ));
  
  EXPECT_TRUE(collision_query_->isFree(
    Eigen::Vector3f(250, 250, 150), 1.0f, std::chrono::steady_clock::now()
  ));
}

TEST_F(CollisionQueryTest, GetDistanceMethod) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  float dist = collision_query_->getDistance(
    Eigen::Vector3f(50, 50, 100), std::chrono::steady_clock::now()
  );
  EXPECT_NEAR(dist, 0.0f, 1e-3f);
  
  dist = collision_query_->getDistance(
    Eigen::Vector3f(70, 50, 100), std::chrono::steady_clock::now()
  );
  EXPECT_NEAR(dist, 10.0f, 1.0f);
}

TEST_F(CollisionQueryTest, GetObstaclesInRadius) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  auto obstacles = collision_query_->getObstaclesInRadius(
    Eigen::Vector3f(50, 50, 100), 30.0f, std::chrono::steady_clock::now()
  );
  
  EXPECT_GE(obstacles.size(), 1);
  
  bool found_static = false;
  for (const auto& obs : obstacles) 
  {
    if (obs.type == HitResult::ObjectType::STATIC_OBSTACLE) 
    {
      found_static = true;
      EXPECT_LT(obs.distance, 30.0f);
    }
  }
  EXPECT_TRUE(found_static);
}

TEST_F(CollisionQueryTest, EarlyExitOptimization) 
{
  collision_query_->setEarlyExitEnabled(true);
  
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(128, 128, 90);
  capsule.p1 = Eigen::Vector3f(128, 128, 100);
  capsule.radius = 5.0f;
  
  auto stats_before = collision_query_->getPerformanceStats();
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  auto stats_after = collision_query_->getPerformanceStats();
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::TERRAIN);
  
  EXPECT_EQ(stats_after.static_checks, stats_before.static_checks);
}

TEST_F(CollisionQueryTest, EarlyExitDisabled) 
{
  collision_query_->setEarlyExitEnabled(false);
  
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(128, 128, 90);
  capsule.p1 = Eigen::Vector3f(128, 128, 100);
  capsule.radius = 5.0f;
  
  auto stats_before = collision_query_->getPerformanceStats();
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  auto stats_after = collision_query_->getPerformanceStats();
  
  EXPECT_TRUE(result.hit);
  
  EXPECT_GT(stats_after.static_checks, stats_before.static_checks);
}

TEST_F(CollisionQueryTest, ZeroRadiusCapsule) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = Eigen::Vector3f(50, 50, 80);
  capsule.p1 = Eigen::Vector3f(50, 50, 120);
  capsule.radius = 0.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::STATIC_OBSTACLE);
}

TEST_F(CollisionQueryTest, DegenerateCapsule) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  Capsule capsule;
  capsule.p0 = capsule.p1 = Eigen::Vector3f(50, 50, 100);
  capsule.radius = 5.0f;
  
  auto result = collision_query_->queryCapsule(capsule, std::chrono::steady_clock::now());
  
  EXPECT_TRUE(result.hit);
  EXPECT_EQ(result.object_type, HitResult::ObjectType::STATIC_OBSTACLE);
}

TEST_F(CollisionQueryTest, JSONExport) 
{
  TerrainConfig terrain_config;
  auto terrain = std::make_shared<terrain::WaveletGrid>(terrain_config);
  ASSERT_TRUE(terrain->loadFromFile(terrain_file_));
  collision_query_->setTerrain(terrain);
  
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  auto corridors = createCorridors();
  collision_query_->setCorridors(corridors);
  
  std::string json_file = (test_dir_ / "collision_export.json").string();
  collision_query_->exportToJSON(json_file);
  
  EXPECT_TRUE(fs::exists(json_file));
  EXPECT_GT(fs::file_size(json_file), 100);
  
  std::ifstream file(json_file);
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  
  EXPECT_NE(content.find("\"components\""), std::string::npos);
  EXPECT_NE(content.find("\"terrain\": true"), std::string::npos);
  EXPECT_NE(content.find("\"static_obstacles\": true"), std::string::npos);
  EXPECT_NE(content.find("\"corridors\": true"), std::string::npos);
}

TEST_F(CollisionQueryTest, ThreadSafety) 
{
  auto static_obs = createStaticObstacles();
  collision_query_->setStaticObstacles(static_obs);
  
  const int num_threads = 4;
  const int queries_per_thread = 100;
  std::vector<std::thread> threads;
  std::atomic<int> hit_count(0);
  
  for (int t = 0; t < num_threads; ++t) 
  {
    threads.emplace_back([this, t, queries_per_thread, &hit_count]() 
    {
      for (int i = 0; i < queries_per_thread; ++i) 
      {
        Capsule c;
        c.p0 = Eigen::Vector3f(50 + t * 0.1f, 50, 95);
        c.p1 = Eigen::Vector3f(50 + t * 0.1f, 50, 105);
        c.radius = 1.0f;
        
        auto result = collision_query_->queryCapsule(c, std::chrono::steady_clock::now());
        if (result.hit) 
        {
          hit_count++;
        }
      }
    });
  }
  
  for (auto& t : threads) 
  {
    t.join();
  }
  
  EXPECT_EQ(hit_count, num_threads * queries_per_thread);
}

}
}
}