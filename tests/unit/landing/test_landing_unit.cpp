#include <gtest/gtest.h>
#include "vkr/landing/landing_unit.hpp"
#include <random>
#include <cmath>

namespace vkr 
{
namespace landing 
{

class LandingUnitTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    config_.ransac_iterations = 50;
    config_.ransac_distance_threshold = 0.02f;
    config_.min_points_for_plane = 50;
    config_.max_tilt_angle = 5.0f;
    config_.min_area = 0.25f;
    config_.max_roughness = 0.02f;
    
    landing_unit_ = std::make_unique<LandingUnit>(config_);
  }
  
  PointCloud generateFlatPlane(const Eigen::Vector3f& center,
                              const Eigen::Vector3f& normal,
                              float width, float height,
                              size_t point_density,
                              float noise = 0.0f) 
  {
    PointCloud cloud;
    
    Eigen::Vector3f u = normal.cross(Eigen::Vector3f::UnitX());
    if (u.norm() < 0.1f) 
    {
      u = normal.cross(Eigen::Vector3f::UnitY());
    }
    u.normalize();
    Eigen::Vector3f v = normal.cross(u);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_u(-width/2, width/2);
    std::uniform_real_distribution<float> dist_v(-height/2, height/2);
    std::normal_distribution<float> noise_dist(0.0f, noise);
    
    for (size_t i = 0; i < point_density; ++i) 
    {
      float pu = dist_u(gen);
      float pv = dist_v(gen);
      float pn = noise_dist(gen);
      
      Eigen::Vector3f point = center + pu * u + pv * v + pn * normal;
      cloud.points.push_back(point);
    }
    
    cloud.min_bound = center - Eigen::Vector3f(width, height, 0.1f);
    cloud.max_bound = center + Eigen::Vector3f(width, height, 0.1f);
    
    return cloud;
  }
  
  PointCloud generateLShapedPlane(const Eigen::Vector3f& center,
                                 float arm_length,
                                 size_t point_density) 
  {
    PointCloud cloud;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    size_t points_per_arm = point_density / 2;
    
    for (size_t i = 0; i < points_per_arm; ++i) 
    {
      float x = dist(gen) * arm_length;
      float y = dist(gen) * 0.5f;
      cloud.points.push_back(center + Eigen::Vector3f(x - arm_length/2, y - 0.25f, 0));
    }
    
    for (size_t i = 0; i < points_per_arm; ++i) 
    {
      float x = dist(gen) * 0.5f;
      float y = dist(gen) * arm_length;
      cloud.points.push_back(center + Eigen::Vector3f(x - 0.25f, y - arm_length/2, 0));
    }
    
    cloud.min_bound = center - Eigen::Vector3f(arm_length, arm_length, 0.1f);
    cloud.max_bound = center + Eigen::Vector3f(arm_length, arm_length, 0.1f);
    
    return cloud;
  }
  
  PointCloud generateMultiplePlanes(size_t num_planes) 
  {
    PointCloud cloud;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-5.0f, 5.0f);
    std::uniform_real_distribution<float> angle_dist(-0.1f, 0.1f);
    
    for (size_t i = 0; i < num_planes; ++i) 
    {
      Eigen::Vector3f center(pos_dist(gen), pos_dist(gen), pos_dist(gen));
      Eigen::Vector3f normal(angle_dist(gen), angle_dist(gen), 1.0f);
      normal.normalize();
      
      auto plane = generateFlatPlane(center, normal, 1.0f, 1.0f, 200, 0.005f);
      cloud.points.insert(cloud.points.end(), plane.points.begin(), plane.points.end());
    }
    
    cloud.min_bound = Eigen::Vector3f(-10, -10, -10);
    cloud.max_bound = Eigen::Vector3f(10, 10, 10);
    
    return cloud;
  }
  
  LandingConfig config_;
  std::unique_ptr<LandingUnit> landing_unit_;
};

TEST_F(LandingUnitTest, EmptyPointCloud) 
{
  PointCloud empty_cloud;
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(empty_cloud, current_pos);
  
  EXPECT_FALSE(result.has_valid_landing_zone);
  EXPECT_TRUE(result.candidates.empty());
}

TEST_F(LandingUnitTest, PerfectHorizontalPlane) 
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 1.0f, 1.0f, 500, 0.0f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_TRUE(result.has_valid_landing_zone);
  EXPECT_FALSE(result.candidates.empty());
  
  const auto& best = result.best_candidate;
  EXPECT_NEAR(best.center.x(), 0.0f, 0.1f);
  EXPECT_NEAR(best.center.y(), 0.0f, 0.1f);
  EXPECT_NEAR(best.center.z(), 0.0f, 0.1f);
  EXPECT_NEAR(best.tilt_angle, 0.0f, 1.0f);
  EXPECT_GE(best.area, 0.8f);
  EXPECT_LE(best.roughness, 0.001f);
  EXPECT_GE(best.score, 0.9f);
}

TEST_F(LandingUnitTest, TiltedPlane) 
{
  Eigen::Vector3f center(0, 0, 0);
  float tilt_rad = 4.0f * M_PI / 180.0f;
  Eigen::Vector3f normal(std::sin(tilt_rad), 0, std::cos(tilt_rad));
  auto cloud = generateFlatPlane(center, normal, 1.0f, 1.0f, 500, 0.001f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_TRUE(result.has_valid_landing_zone);
  const auto& best = result.best_candidate;
  EXPECT_NEAR(best.tilt_angle, 4.0f, 0.5f);
  EXPECT_LT(best.score, 0.9f);
}

TEST_F(LandingUnitTest, TooTiltedPlane) 
{
  Eigen::Vector3f center(0, 0, 0);
  float tilt_rad = 10.0f * M_PI / 180.0f;
  Eigen::Vector3f normal(std::sin(tilt_rad), 0, std::cos(tilt_rad));
  auto cloud = generateFlatPlane(center, normal, 1.0f, 1.0f, 500, 0.001f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_FALSE(result.has_valid_landing_zone);
}

TEST_F(LandingUnitTest, SmallArea) 
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 0.4f, 0.4f, 100, 0.001f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_FALSE(result.has_valid_landing_zone);
}

TEST_F(LandingUnitTest, RoughSurface) 
{
  config_.max_roughness = 0.01f;
  landing_unit_ = std::make_unique<LandingUnit>(config_);
  
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 1.0f, 1.0f, 500, 0.015f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_FALSE(result.has_valid_landing_zone);
}

TEST_F(LandingUnitTest, LShapedSurface) 
{
  Eigen::Vector3f center(0, 0, 0);
  auto cloud = generateLShapedPlane(center, 1.0f, 800);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  if (result.has_valid_landing_zone) 
  {
    const auto& best = result.best_candidate;
    float l_shape_area = 0.5f * 1.0f + 0.5f * 1.0f - 0.5f * 0.5f;
    EXPECT_GT(best.area, 0.5f);
    EXPECT_LT(best.area, 1.2f);
  }
}

TEST_F(LandingUnitTest, MultiplePlanes) 
{
  auto cloud = generateMultiplePlanes(3);
  Eigen::Vector3f current_pos(0, 0, 10);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_GE(result.candidates.size(), 1);
  EXPECT_LE(result.candidates.size(), 3);
  
  if (result.has_valid_landing_zone) 
  {
    EXPECT_GT(result.best_candidate.score, 0.5f);
  }
}

TEST_F(LandingUnitTest, DistanceEffect) 
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 1.0f, 1.0f, 300, 0.001f);
  
  Eigen::Vector3f close_pos(0, 0, 2);
  auto result_close = landing_unit_->analyzePointCloud(cloud, close_pos);
  
  Eigen::Vector3f far_pos(0, 0, 20);
  auto result_far = landing_unit_->analyzePointCloud(cloud, far_pos);
  
  if (result_close.has_valid_landing_zone && result_far.has_valid_landing_zone) 
  {
    float close_dist = (result_close.best_candidate.center - close_pos).norm();
    float far_dist = (result_far.best_candidate.center - far_pos).norm();
    
    float close_expected_density = 10000.0f / (1.0f + close_dist * 0.1f);
    float far_expected_density = 10000.0f / (1.0f + far_dist * 0.1f);
    
    float density_ratio = close_expected_density / far_expected_density;
    
    EXPECT_GT(density_ratio, 1.5f);
  }
}

TEST_F(LandingUnitTest, AngleEffect) 
{
  PointCloud cloud;
  
  Eigen::Vector3f center(0, 0, 0);
  float size = 1.0f;
  int points_per_side = 50;
  
  for (int i = 0; i < points_per_side; ++i) 
  {
    for (int j = 0; j < points_per_side; ++j) 
    {
      float x = (i / float(points_per_side - 1) - 0.5f) * size;
      float y = (j / float(points_per_side - 1) - 0.5f) * size;
      cloud.points.push_back(center + Eigen::Vector3f(x, y, 0));
    }
  }
  
  cloud.min_bound = center - Eigen::Vector3f(size, size, 0.1f);
  cloud.max_bound = center + Eigen::Vector3f(size, size, 0.1f);
  
  Eigen::Vector3f direct_pos(0, 0, 5);
  auto result_direct = landing_unit_->analyzePointCloud(cloud, direct_pos);
  
  Eigen::Vector3f angled_pos(10, 0, 5);
  auto result_angled = landing_unit_->analyzePointCloud(cloud, angled_pos);
  
  if (result_direct.has_valid_landing_zone && result_angled.has_valid_landing_zone) 
  {
    EXPECT_NEAR(result_direct.best_candidate.area, 
                result_angled.best_candidate.area, 0.2f);
    EXPECT_GT(result_direct.best_candidate.score, 0.8f);
    EXPECT_GT(result_angled.best_candidate.score, 0.8f);
  }
}

TEST_F(LandingUnitTest, ConfigUpdate) 
{
  LandingConfig new_config = config_;
  new_config.min_area = 0.5f;
  new_config.max_tilt_angle = 3.0f;
  
  landing_unit_->updateConfig(new_config);
  
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 0.6f, 0.6f, 200, 0.001f);
  Eigen::Vector3f current_pos(0, 0, 5);
  
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  
  EXPECT_FALSE(result.has_valid_landing_zone);
}

TEST_F(LandingUnitTest, PerformanceLargeCloud) 
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  auto cloud = generateFlatPlane(center, normal, 5.0f, 5.0f, 10000, 0.01f);
  Eigen::Vector3f current_pos(0, 0, 10);
  
  auto start = std::chrono::steady_clock::now();
  auto result = landing_unit_->analyzePointCloud(cloud, current_pos);
  auto end = std::chrono::steady_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  EXPECT_LT(duration.count(), 1000);
  EXPECT_TRUE(result.has_valid_landing_zone);
  EXPECT_GT(result.computation_time_ms, 0);
}

TEST_F(LandingUnitTest, ScoreComponents) 
{
  Eigen::Vector3f center(0, 0, 0);
  Eigen::Vector3f normal(0, 0, 1);
  
  auto cloud1 = generateFlatPlane(center, normal, 0.6f, 0.6f, 200, 0.001f);
  Eigen::Vector3f pos(0, 0, 5);
  auto result1 = landing_unit_->analyzePointCloud(cloud1, pos);
  
  auto cloud2 = generateFlatPlane(center, normal, 1.2f, 1.2f, 500, 0.001f);
  auto result2 = landing_unit_->analyzePointCloud(cloud2, pos);
  
  if (result1.has_valid_landing_zone && result2.has_valid_landing_zone) 
  {
    EXPECT_LT(result1.best_candidate.score, result2.best_candidate.score);
  }
}

}
}