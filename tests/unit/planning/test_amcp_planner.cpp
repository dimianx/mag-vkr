#include <gtest/gtest.h>
#include "vkr/planning/amcp_planner.hpp"
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/terrain/qwg_builder.hpp"
#include "vkr/geometry/collision/collision_query.hpp"
#include "vkr/geometry/corridors/spatial_corridor_map.hpp"
#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include <memory>
#include <chrono>
#include <filesystem>
#include <random>
#include <gdal_priv.h>

namespace vkr 
{
namespace planning 
{

class AMCPlannerTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.planning.coarse_grid_size = 20.0f;
    config_.planning.fine_grid_size = 5.0f;
    config_.planning.initial_epsilon = 2.5f;
    config_.planning.height_penalty_weight = 10.0f;
    config_.planning.max_runtime_ms = 500.0f;
    config_.planning.safe_altitude = 50.0f;
    config_.planning.min_altitude = 30.0f;
    config_.planning.min_clearance = 10.0f;
    
    config_.geometry.corridor_radius = 5.0f;
    config_.geometry.grid_cell_size = 40.0f;
    
    config_.terrain.wavelet_levels = 4;
    config_.terrain.page_size = 4096;
    config_.terrain.max_cache_size = 10;
    
    planner_ = std::make_unique<AMCPlanner>(config_);
    
    collision_ = std::make_shared<geometry::collision::CollisionQuery>(config_.geometry);
    planner_->setCollisionQuery(collision_);
    
    test_dir_ = std::filesystem::current_path() / "tests" / "data" / "temp_planner";
    std::filesystem::create_directories(test_dir_);
    
    GDALAllRegister();
  }
  
  void TearDown() override
  {
    if (std::filesystem::exists(test_dir_)) 
    {
      std::filesystem::remove_all(test_dir_);
    }
  }
  
  PlanRequest createRequestWithBounds(const Eigen::Vector3f& start, 
                                      const Eigen::Vector3f& goal,
                                      const geometry::BoundingBox& bounds) 
  {
    PlanRequest request;
    request.start = start;
    request.goal = goal;
    request.uav_id = 1;
    request.max_runtime_ms = 100.0f;
    request.setWorldBounds(bounds);
    return request;
  }
  
  std::shared_ptr<terrain::WaveletGrid> createTestTerrain() 
  {
    constexpr size_t size = 256;
    std::string tif_file = (test_dir_ / "test_terrain.tif").string();
    
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) 
    {
      throw std::runtime_error("GTiff driver not available");
    }
    
    GDALDataset* dataset = driver->Create(tif_file.c_str(), size, size, 
                                         1, GDT_Float32, nullptr);
    if (!dataset) 
    {
      throw std::runtime_error("Failed to create GeoTIFF");
    }
    
    double geotransform[6] = {0, 5, 0, 1280, 0, -5};
    dataset->SetGeoTransform(geotransform);
    
    std::vector<float> buffer(size * size);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise_dist(-5.0f, 5.0f);
    
    for (size_t y = 0; y < size; ++y) 
    {
      for (size_t x = 0; x < size; ++x) 
      {
        float base = 100.0f;
        float hill1 = 50.0f * std::exp(-((x-64.f)*(x-64.f) + (y-64.f)*(y-64.f)) / 1000.0f);
        float hill2 = 30.0f * std::exp(-((x-192.f)*(x-192.f) + (y-192.f)*(y-192.f)) / 800.0f);
        float noise = noise_dist(rng);
        buffer[y * size + x] = base + hill1 + hill2 + noise;
      }
    }
    
    GDALRasterBand* band = dataset->GetRasterBand(1);
    band->RasterIO(GF_Write, 0, 0, size, size, 
                   buffer.data(), size, size, GDT_Float32, 0, 0);
    band->SetNoDataValue(-9999.0);
    
    GDALFlushCache(dataset);
    GDALClose(dataset);
    
    std::string qwg_file = (test_dir_ / "test_terrain.qwg").string();
    
    terrain::QWGBuilder builder(config_.terrain);
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(tif_file, qwg_file, error_message);
    if (!success) 
    {
      throw std::runtime_error("Failed to create QWG: " + error_message);
    }
    
    auto terrain = std::make_shared<terrain::WaveletGrid>(config_.terrain);
    if (!terrain->loadFromFile(qwg_file)) 
    {
      throw std::runtime_error("Failed to load QWG file");
    }
    
    return terrain;
  }
  
  VkrConfig config_;
  std::unique_ptr<AMCPlanner> planner_;
  std::shared_ptr<geometry::collision::CollisionQuery> collision_;
  std::filesystem::path test_dir_;
};

TEST_F(AMCPlannerTest, BasicPathPlanning) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(1100, 1100, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(1000, 1000, 100),
    world_bounds
  );
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  EXPECT_FALSE(result.path.empty());
  EXPECT_GE(result.path.size(), 2u);
  
  EXPECT_LT((result.path.front() - request.start).norm(), GEOM_EPS) 
    << "Path should start at exact start position";
  EXPECT_LT((result.path.back() - request.goal).norm(), GEOM_EPS)
    << "Path should end at exact goal position";
  
  for (size_t i = 1; i < result.path.size(); ++i) 
  {
    float step = (result.path[i] - result.path[i-1]).norm();
    EXPECT_LE(step, config_.planning.fine_grid_size * std::sqrt(3.0f) * 1.5f);
  }
}

TEST_F(AMCPlannerTest, PlanningWithObstacles) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(1100, 1100, 300);
  
  auto bst = std::make_shared<geometry::static_obstacles::BoundingSphereTree>();
  
  std::vector<geometry::static_obstacles::BoundingSphereTree::Face> faces;
  for (float z = 0; z < 200; z += 10) 
  {
    geometry::static_obstacles::BoundingSphereTree::Face face;
    face.vertices[0] = Eigen::Vector3f(490, 490, z);
    face.vertices[1] = Eigen::Vector3f(510, 490, z);
    face.vertices[2] = Eigen::Vector3f(500, 510, z);
    face.normal = Eigen::Vector3f(0, 0, 1);
    face.object_id = 1;
    faces.push_back(face);
  }
  
  bst->build(faces);
  collision_->setStaticObstacles(bst);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(1000, 1000, 100),
    world_bounds
  );
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  for (const auto& pos : result.path) 
  {
    float dist_to_obstacle = (pos.head<2>() - Eigen::Vector2f(500, 500)).norm();
    EXPECT_GT(dist_to_obstacle, config_.geometry.corridor_radius);
  }
}

TEST_F(AMCPlannerTest, PlanningWithCorridors) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(1100, 1100, 200);
  
  auto corridors = std::make_shared<geometry::corridors::SpatialCorridorMap>(config_.geometry);
  collision_->setCorridors(corridors);
  planner_->setCorridors(corridors);
  
  std::vector<Eigen::Vector3f> other_path = 
  {
    Eigen::Vector3f(400, 0, 100),
    Eigen::Vector3f(400, 1000, 100),
    Eigen::Vector3f(600, 1000, 100),
    Eigen::Vector3f(600, 0, 100)
  };
  
  CorridorId other_corridor = corridors->createCorridor(2, other_path, 
                                                        config_.geometry.corridor_radius);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 500, 100),
    Eigen::Vector3f(1000, 500, 100),
    world_bounds
  );
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  for (const auto& pos : result.path) 
  {
    EXPECT_FALSE(corridors->isInsideOtherCorridor(pos, request.uav_id));
  }
}


TEST_F(AMCPlannerTest, AsyncPlanning) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(1100, 1100, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(1000, 1000, 100),
    world_bounds
  );
  
  auto future = planner_->planAsync(request);
  
  auto status = future.wait_for(std::chrono::milliseconds(1000));
  EXPECT_EQ(status, std::future_status::ready);
  
  auto result = future.get();
  EXPECT_EQ(result.status, Status::SUCCESS);
}

TEST_F(AMCPlannerTest, TimeoutHandling) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-1000, -1000, 0);
  world_bounds.max = Eigen::Vector3f(11000, 11000, 200);
  
  PlanRequest request;
  request.start = Eigen::Vector3f(0, 0, 100);
  request.goal = Eigen::Vector3f(10000, 10000, 100);
  request.uav_id = 1;
  request.max_runtime_ms = .001f;
  request.setWorldBounds(world_bounds);
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::TIMEOUT);
}

TEST_F(AMCPlannerTest, DynamicReplanning) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(1100, 100, 200);
  
  auto corridors = std::make_shared<geometry::corridors::SpatialCorridorMap>(config_.geometry);
  collision_->setCorridors(corridors);
  planner_->setCorridors(corridors);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(1000, 0, 100),
    world_bounds
  );
  
  auto initial_result = planner_->plan(request);
  EXPECT_EQ(initial_result.status, Status::SUCCESS);
  
  CorridorId own_corridor = corridors->createCorridor(1, initial_result.path, 
                                                      config_.geometry.corridor_radius);
  
  geometry::corridors::CorridorFreeEvent event;
  event.corridor_id = own_corridor;
  event.segment_id = 1;
  event.timestamp = std::chrono::steady_clock::now();
  
  planner_->pushEvent(event);
  
  auto new_result = planner_->plan(request);
  EXPECT_EQ(new_result.status, Status::SUCCESS);
}

TEST_F(AMCPlannerTest, MultiResolutionPlanning) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-500, -500, 0);
  world_bounds.max = Eigen::Vector3f(2500, 2500, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(2000, 2000, 100),
    world_bounds
  );
  request.max_runtime_ms = 100.0f;
  
  auto start_time = std::chrono::steady_clock::now();
  auto result = planner_->plan(request);
  auto elapsed = std::chrono::steady_clock::now() - start_time;
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  EXPECT_LT(result.planning_time.count(), request.max_runtime_ms);
}

TEST_F(AMCPlannerTest, StatisticsCollection) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(600, 600, 200);
  
  for (int i = 0; i < 5; ++i) 
  {
    auto request = createRequestWithBounds(
      Eigen::Vector3f(i * 20.0f, 0, 100),
      Eigen::Vector3f(i * 20.0f + 400.0f, 400, 100),
      world_bounds
    );
    
    planner_->plan(request);
  }
  
  auto stats = planner_->getStats();
  
  EXPECT_EQ(stats.total_plans, 5u);
  EXPECT_GT(stats.successful_plans, 0u);
  EXPECT_GT(stats.average_planning_time_ms, 0.0f);
  EXPECT_GT(stats.average_path_length, 0.0f);
}

TEST_F(AMCPlannerTest, ComplexScenario) 
{
  auto terrain = createTestTerrain();
  planner_->setTerrain(terrain);
  collision_->setTerrain(terrain);
  
  auto bst = std::make_shared<geometry::static_obstacles::BoundingSphereTree>();
  std::vector<geometry::static_obstacles::BoundingSphereTree::Face> faces;
  
  for (float z = 0; z < 100; z += 10) 
  {
    geometry::static_obstacles::BoundingSphereTree::Face face;
    face.vertices[0] = Eigen::Vector3f(500, 500, z);
    face.vertices[1] = Eigen::Vector3f(550, 500, z);
    face.vertices[2] = Eigen::Vector3f(525, 550, z);
    face.normal = Eigen::Vector3f(0, 0, 1);
    face.object_id = 1;
    faces.push_back(face);
  }
  
  bst->build(faces);
  collision_->setStaticObstacles(bst);
  
  auto corridors = std::make_shared<geometry::corridors::SpatialCorridorMap>(config_.geometry);
  collision_->setCorridors(corridors);
  planner_->setCorridors(corridors);
  
  std::vector<Eigen::Vector3f> uav2_path = 
  {
    Eigen::Vector3f(200, 0, 150),
    Eigen::Vector3f(200, 1000, 150)
  };
  corridors->createCorridor(2, uav2_path, config_.geometry.corridor_radius);
  
  PlanRequest request;
  request.start = Eigen::Vector3f(50, 50, 200);
  request.goal = Eigen::Vector3f(950, 950, 200);
  request.uav_id = 1;
  request.max_runtime_ms = 500.0f;
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  for (const auto& pos : result.path) 
  {
    if ((pos - request.start).norm() < GEOM_EPS || (pos - request.goal).norm() < GEOM_EPS) 
    {
      continue;
    }
    
    float ground_height = terrain->getHeight(pos.x(), pos.y());
    EXPECT_GE(pos.z() - ground_height, config_.planning.min_clearance);
    
    EXPECT_FALSE(corridors->isInsideOtherCorridor(pos, request.uav_id));
    
    float dist_to_building = (pos.head<2>() - Eigen::Vector2f(525, 525)).norm();
    if (pos.z() < 100) 
    {
      EXPECT_GT(dist_to_building, 50.0f);
    }
  }
}

TEST_F(AMCPlannerTest, EdgeCostUpdates) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(600, 600, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(500, 500, 100),
    world_bounds
  );
  
  auto initial = planner_->plan(request);
  EXPECT_EQ(initial.status, Status::SUCCESS);
  
  if (initial.path.size() > 2) 
  {
    NodeId from = planner_->getStats().current_open_list_size;
    NodeId to = from + 1;
    
    planner_->updateEdgeCost(from, to, 1000.0f);
    
    auto updated = planner_->plan(request);
    EXPECT_EQ(updated.status, Status::SUCCESS);
  }
}

TEST_F(AMCPlannerTest, ClearAndReset) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(600, 600, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(500, 500, 100),
    world_bounds
  );
  
  planner_->plan(request);
  
  auto stats_before = planner_->getStats();
  EXPECT_GT(stats_before.total_plans, 0u);
  
  planner_->clear();
  
  auto stats_after = planner_->getStats();
  EXPECT_EQ(stats_after.total_plans, 0u);
  
  auto result = planner_->plan(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
}

TEST_F(AMCPlannerTest, WorldBoundsFromRequest) 
{
  geometry::BoundingBox small_bounds;
  small_bounds.min = Eigen::Vector3f(0, 0, 50);
  small_bounds.max = Eigen::Vector3f(200, 200, 150);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(50, 50, 100),
    Eigen::Vector3f(150, 150, 100),
    small_bounds
  );
  
  auto result = planner_->plan(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
  
  geometry::BoundingBox large_bounds;
  large_bounds.min = Eigen::Vector3f(-1000, -1000, 0);
  large_bounds.max = Eigen::Vector3f(2000, 2000, 500);
  
  request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(1500, 1500, 100),
    large_bounds
  );
  
  result = planner_->plan(request);
  EXPECT_EQ(result.status, Status::SUCCESS);
}

TEST_F(AMCPlannerTest, AutomaticBoundsExpansion) 
{
  PlanRequest request;
  request.start = Eigen::Vector3f(0, 0, 100);
  request.goal = Eigen::Vector3f(1000, 1000, 100);
  request.uav_id = 1;
  
  auto result = planner_->plan(request);
  
  EXPECT_EQ(result.status, Status::SUCCESS);
  EXPECT_FALSE(result.path.empty());
}

TEST_F(AMCPlannerTest, JSONExport) 
{
  geometry::BoundingBox world_bounds;
  world_bounds.min = Eigen::Vector3f(-100, -100, 0);
  world_bounds.max = Eigen::Vector3f(600, 600, 200);
  
  auto request = createRequestWithBounds(
    Eigen::Vector3f(0, 0, 100),
    Eigen::Vector3f(500, 500, 100),
    world_bounds
  );
  
  planner_->plan(request);
  
  std::string filename = (test_dir_ / "planner_export.json").string();
  planner_->exportToJSON(filename);
  
  EXPECT_TRUE(std::filesystem::exists(filename));
  EXPECT_GT(std::filesystem::file_size(filename), 100u);
}

}
}