#include <gtest/gtest.h>
#include "vkr/geometry/dynamic_obstacles/swept_obb.hpp"
#include "vkr/geometry/intersections.hpp"
#include <chrono>
#include <cmath>

namespace vkr 
{
namespace geometry 
{
namespace dynamic_obstacles 
{

class SweptOBBTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    basic_obb_.base_obb.center = Eigen::Vector3f(0, 0, 0);
    basic_obb_.base_obb.half_extents = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
    basic_obb_.base_obb.orientation = Eigen::Matrix3f::Identity();
    basic_obb_.object_id = 42;
    
    basic_obb_.t0 = std::chrono::steady_clock::now();
    basic_obb_.t1 = basic_obb_.t0 + std::chrono::seconds(10);
    
    basic_obb_.position_coeffs = 
    {
      Eigen::Vector3f(0, 0, 0),
      Eigen::Vector3f(1, 0, 0)
    };
  }
  
  SweptOBB basic_obb_;
};

TEST_F(SweptOBBTest, BasicConstruction) 
{
  SweptOBB obb;
  EXPECT_EQ(obb.object_id, 0);
  EXPECT_TRUE(obb.position_coeffs.empty());
  EXPECT_TRUE(obb.rotation_coeffs.empty());
  EXPECT_TRUE(obb.time_coeffs.empty());
}

TEST_F(SweptOBBTest, OBBGetCorners) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(1, 2, 3);
  obb.half_extents = Eigen::Vector3f(0.5f, 1.0f, 1.5f);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  auto corners = obb.getCorners();
  EXPECT_EQ(corners.size(), 8);
  
  for (const auto& corner : corners) 
  {
    Eigen::Vector3f local = corner - obb.center;
    EXPECT_LE(std::abs(local.x()), obb.half_extents.x() + 1e-6f);
    EXPECT_LE(std::abs(local.y()), obb.half_extents.y() + 1e-6f);
    EXPECT_LE(std::abs(local.z()), obb.half_extents.z() + 1e-6f);
  }
}

TEST_F(SweptOBBTest, OBBGetAABB) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 1, 1);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  auto aabb = obb.getAABB();
  
  EXPECT_NEAR(aabb.min.x(), -1.0f, 1e-6f);
  EXPECT_NEAR(aabb.max.x(), 1.0f, 1e-6f);
  EXPECT_NEAR(aabb.min.y(), -1.0f, 1e-6f);
  EXPECT_NEAR(aabb.max.y(), 1.0f, 1e-6f);
  EXPECT_NEAR(aabb.min.z(), -1.0f, 1e-6f);
  EXPECT_NEAR(aabb.max.z(), 1.0f, 1e-6f);
}

TEST_F(SweptOBBTest, OBBGetAABB_Rotated) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(2, 1, 0.5f);
  
  float angle = M_PI / 4.0f;
  obb.orientation = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  
  auto aabb = obb.getAABB();
  
  float expected_xy = (2.0f + 1.0f) / std::sqrt(2.0f);
  EXPECT_NEAR(aabb.min.x(), -expected_xy, 1e-5f);
  EXPECT_NEAR(aabb.max.x(), expected_xy, 1e-5f);
  EXPECT_NEAR(aabb.min.y(), -expected_xy, 1e-5f);
  EXPECT_NEAR(aabb.max.y(), expected_xy, 1e-5f);
  EXPECT_NEAR(aabb.min.z(), -0.5f, 1e-6f);
  EXPECT_NEAR(aabb.max.z(), 0.5f, 1e-6f);
}

TEST_F(SweptOBBTest, IsActiveAtTime) 
{
  auto now = std::chrono::steady_clock::now();
  auto t0 = now;
  auto t1 = now + std::chrono::seconds(10);
  
  basic_obb_.t0 = t0;
  basic_obb_.t1 = t1;
  
  EXPECT_FALSE(basic_obb_.isActiveAtTime(t0 - std::chrono::seconds(1)));
  
  EXPECT_TRUE(basic_obb_.isActiveAtTime(t0));
  
  EXPECT_TRUE(basic_obb_.isActiveAtTime(t0 + std::chrono::seconds(5)));
  
  EXPECT_TRUE(basic_obb_.isActiveAtTime(t1));
  
  EXPECT_FALSE(basic_obb_.isActiveAtTime(t1 + std::chrono::seconds(1)));
}

TEST_F(SweptOBBTest, GetOBBAtTime_Linear) 
{
  basic_obb_.computeBounds();
  
  auto obb_start = basic_obb_.getOBBAtTime(basic_obb_.t0);
  EXPECT_NEAR(obb_start.center.x(), 0.0f, 1e-6f);
  EXPECT_NEAR(obb_start.center.y(), 0.0f, 1e-6f);
  EXPECT_NEAR(obb_start.center.z(), 0.0f, 1e-6f);
  
  auto mid_time = basic_obb_.t0 + std::chrono::milliseconds(5000);
  auto obb_mid = basic_obb_.getOBBAtTime(mid_time);
  EXPECT_NEAR(obb_mid.center.x(), 0.5f, 1e-2f);
  
  auto obb_end = basic_obb_.getOBBAtTime(basic_obb_.t1);
  EXPECT_NEAR(obb_end.center.x(), 1.0f, 1e-2f);
}

TEST_F(SweptOBBTest, GetOBBAtTime_WithRotation) 
{
  basic_obb_.rotation_coeffs = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0, 0, M_PI/2)
  };
  basic_obb_.computeBounds();
  
  auto obb_start = basic_obb_.getOBBAtTime(basic_obb_.t0);
  auto obb_end = basic_obb_.getOBBAtTime(basic_obb_.t1);
  
  EXPECT_FALSE(obb_start.orientation.isApprox(obb_end.orientation));
  
  Eigen::Vector3f x_axis_end = obb_end.orientation.col(0);
  EXPECT_NEAR(x_axis_end.x(), 0.0f, 1e-2f);
  EXPECT_NEAR(std::abs(x_axis_end.y()), 1.0f, 1e-2f);
}

TEST_F(SweptOBBTest, DistanceToPoint) 
{
  basic_obb_.position_coeffs = {Eigen::Vector3f(0, 0, 0)};
  basic_obb_.computeBounds();
  
  Eigen::Vector3f point_outside(2, 0, 0);
  float dist = basic_obb_.distanceToPoint(point_outside, basic_obb_.t0);
  EXPECT_NEAR(dist, 1.0f, 1e-2f);
  
  Eigen::Vector3f point_inside(0.5f, 0, 0);
  dist = basic_obb_.distanceToPoint(point_inside, basic_obb_.t0);
  EXPECT_LE(dist, 0.0f);
  
  Eigen::Vector3f point_surface(1.0f, 0, 0);
  dist = basic_obb_.distanceToPoint(point_surface, basic_obb_.t0);
  EXPECT_NEAR(dist, 0.0f, 1e-2f);
}

TEST_F(SweptOBBTest, ComputeBounds_Static) 
{
  basic_obb_.position_coeffs = {Eigen::Vector3f(1, 2, 3)};
  basic_obb_.rotation_coeffs.clear();
  basic_obb_.computeBounds();
  
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.min.x(), 0.0f, 1e-2f);
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.max.x(), 2.0f, 1e-2f);
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.min.y(), 1.5f, 1e-2f);
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.max.y(), 2.5f, 1e-2f);
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.min.z(), 2.75f, 1e-2f);
  EXPECT_NEAR(basic_obb_.cached.spatial_aabb.max.z(), 3.25f, 1e-2f);
}

TEST_F(SweptOBBTest, ComputeBounds_Moving) 
{
  basic_obb_.position_coeffs = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(10, 0, 0)
  };
  basic_obb_.computeBounds();
  
  EXPECT_LE(basic_obb_.cached.spatial_aabb.min.x(), -1.0f);
  EXPECT_GE(basic_obb_.cached.spatial_aabb.max.x(), 11.0f);
}

TEST_F(SweptOBBTest, ComputeBounds_Rotating) 
{
  basic_obb_.position_coeffs = {Eigen::Vector3f(0, 0, 0)};
  basic_obb_.rotation_coeffs = 
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0, 0, M_PI/2)
  };
  basic_obb_.computeBounds();
  
  EXPECT_LE(basic_obb_.cached.spatial_aabb.min.x(), -1.0f);
  EXPECT_GE(basic_obb_.cached.spatial_aabb.max.x(), 1.0f);
  EXPECT_LE(basic_obb_.cached.spatial_aabb.min.y(), -1.0f);
  EXPECT_GE(basic_obb_.cached.spatial_aabb.max.y(), 1.0f);
}

TEST_F(SweptOBBTest, IntersectOBBOBB_Separated) 
{
  OBB obb1, obb2;
  obb1.center = Eigen::Vector3f(0, 0, 0);
  obb1.half_extents = Eigen::Vector3f(1, 1, 1);
  obb1.orientation = Eigen::Matrix3f::Identity();
  
  obb2.center = Eigen::Vector3f(3, 0, 0);
  obb2.half_extents = Eigen::Vector3f(1, 1, 1);
  obb2.orientation = Eigen::Matrix3f::Identity();
  
  EXPECT_FALSE(intersectOBBOBB(obb1, obb2));
}

TEST_F(SweptOBBTest, IntersectOBBOBB_Overlapping) 
{
  OBB obb1, obb2;
  obb1.center = Eigen::Vector3f(0, 0, 0);
  obb1.half_extents = Eigen::Vector3f(1, 1, 1);
  obb1.orientation = Eigen::Matrix3f::Identity();
  
  obb2.center = Eigen::Vector3f(1.5f, 0, 0);
  obb2.half_extents = Eigen::Vector3f(1, 1, 1);
  obb2.orientation = Eigen::Matrix3f::Identity();
  
  EXPECT_TRUE(intersectOBBOBB(obb1, obb2));
}

TEST_F(SweptOBBTest, IntersectOBBOBB_Rotated) 
{
  OBB obb1, obb2;
  obb1.center = Eigen::Vector3f(0, 0, 0);
  obb1.half_extents = Eigen::Vector3f(2, 0.1f, 0.1f);
  obb1.orientation = Eigen::Matrix3f::Identity();
  
  obb2.center = Eigen::Vector3f(0, 0, 0);
  obb2.half_extents = Eigen::Vector3f(2, 0.1f, 0.1f);
  obb2.orientation = Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitZ()).toRotationMatrix();
  
  EXPECT_TRUE(intersectOBBOBB(obb1, obb2));
  
  obb2.center = Eigen::Vector3f(0, 3, 0);
  EXPECT_FALSE(intersectOBBOBB(obb1, obb2));
}

TEST_F(SweptOBBTest, IntersectPointOBB) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 2, 3);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  EXPECT_TRUE(intersectPointOBB(Eigen::Vector3f(0, 0, 0), obb));
  EXPECT_TRUE(intersectPointOBB(Eigen::Vector3f(0.5f, 1.5f, 2.5f), obb));
  
  EXPECT_TRUE(intersectPointOBB(Eigen::Vector3f(1, 0, 0), obb));
  EXPECT_TRUE(intersectPointOBB(Eigen::Vector3f(0, 2, 0), obb));
  EXPECT_TRUE(intersectPointOBB(Eigen::Vector3f(0, 0, 3), obb));
  
  EXPECT_FALSE(intersectPointOBB(Eigen::Vector3f(1.1f, 0, 0), obb));
  EXPECT_FALSE(intersectPointOBB(Eigen::Vector3f(0, 2.1f, 0), obb));
  EXPECT_FALSE(intersectPointOBB(Eigen::Vector3f(0, 0, 3.1f), obb));
}

TEST_F(SweptOBBTest, IntersectSegmentOBB) 
{
  OBB obb;
  obb.center = Eigen::Vector3f(0, 0, 0);
  obb.half_extents = Eigen::Vector3f(1, 1, 1);
  obb.orientation = Eigen::Matrix3f::Identity();
  
  LineSegment seg1;
  seg1.start = Eigen::Vector3f(-2, 0, 0);
  seg1.end = Eigen::Vector3f(2, 0, 0);
  EXPECT_TRUE(intersectSegmentOBB(seg1, obb));
  
  LineSegment seg2;
  seg2.start = Eigen::Vector3f(-0.5f, 0, 0);
  seg2.end = Eigen::Vector3f(0.5f, 0, 0);
  EXPECT_TRUE(intersectSegmentOBB(seg2, obb));
  
  LineSegment seg3;
  seg3.start = Eigen::Vector3f(2, 0, 0);
  seg3.end = Eigen::Vector3f(3, 0, 0);
  EXPECT_FALSE(intersectSegmentOBB(seg3, obb));
  
  LineSegment seg4;
  seg4.start = Eigen::Vector3f(0, 0, 0);
  seg4.end = Eigen::Vector3f(2, 2, 2);
  EXPECT_TRUE(intersectSegmentOBB(seg4, obb));
}

}
}
}