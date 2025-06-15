#include <gtest/gtest.h>
#include "vkr/geometry/intersections.hpp"
#include "vkr/geometry/primitives.hpp"
#include <random>
#include <vector>
#include <chrono>

namespace vkr 
{
namespace geometry 
{

class IntersectionTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    rng_.seed(42);
  }
  
  std::mt19937 rng_;
  std::uniform_real_distribution<float> pos_dist_{-100.0f, 100.0f};
  std::uniform_real_distribution<float> radius_dist_{0.1f, 10.0f};
  std::uniform_real_distribution<float> unit_dist_{-1.0f, 1.0f};
};

TEST_F(IntersectionTest, SphereSphereIntersection) 
{
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1.0f};
    Sphere b{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
  
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1.0f};
    Sphere b{Eigen::Vector3f(2, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
  
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 2.0f};
    Sphere b{Eigen::Vector3f(1, 0, 0), 2.0f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
  
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1.0f};
    Sphere b{Eigen::Vector3f(3, 0, 0), 1.0f};
    EXPECT_FALSE(intersectSphereSphere(a, b));
  }
  
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 5.0f};
    Sphere b{Eigen::Vector3f(1, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
}

TEST_F(IntersectionTest, SegmentSphereIntersection) 
{
  {
    LineSegment seg{Eigen::Vector3f(-2, 0, 0), Eigen::Vector3f(2, 0, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSegmentSphere(seg, sph));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(-2, 1, 0), Eigen::Vector3f(2, 1, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSegmentSphere(seg, sph));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(-2, 2, 0), Eigen::Vector3f(2, 2, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_FALSE(intersectSegmentSphere(seg, sph));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 0, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSegmentSphere(seg, sph));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(-0.5f, 0, 0), Eigen::Vector3f(0.5f, 0, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 2.0f};
    EXPECT_TRUE(intersectSegmentSphere(seg, sph));
  }
}

TEST_F(IntersectionTest, CapsuleCapsuleIntersection) 
{
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(1, 1, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, 1, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(3, 0, 0), Eigen::Vector3f(4, 0, 0), 0.5f};
    EXPECT_FALSE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.3f};
    Capsule b{Eigen::Vector3f(0.5f, 0.5f, 1), Eigen::Vector3f(0.5f, 0.5f, -1), 0.3f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
}

TEST_F(IntersectionTest, DegenerateCapsules) 
{
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0), 1.0f};
    Capsule b{Eigen::Vector3f(1.5f, 0, 0), Eigen::Vector3f(1.5f, 0, 0), 1.0f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0), 1.0f};
    Capsule b{Eigen::Vector3f(0.5f, 0, 0), Eigen::Vector3f(2, 0, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0), 1.0f};
    Capsule b{Eigen::Vector3f(3, 0, 0), Eigen::Vector3f(3, 0, 0), 1.0f};
    EXPECT_FALSE(intersectCapsuleCapsule(a, b));
  }
}

TEST_F(IntersectionTest, CollinearCapsules) 
{
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(3, 0, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(2, 0, 0), Eigen::Vector3f(3, 0, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(3, 0, 0), Eigen::Vector3f(4, 0, 0), 0.5f};
    EXPECT_FALSE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(4, 0, 0), 1.0f};
    Capsule b{Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(2, 0, 0), 0.3f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
}

TEST_F(IntersectionTest, ThinCapsules) 
{
  const float epsilon = 1e-6f;
  
  {
    Capsule a{Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(1, 0, 0), epsilon};
    Capsule b{Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, 1, 0), epsilon};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 0, 0), epsilon};
    Capsule b{Eigen::Vector3f(1, -1, 0), Eigen::Vector3f(1, 1, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
}

TEST_F(IntersectionTest, TangentIntersections) 
{
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1.0f};
    Sphere b{Eigen::Vector3f(2, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
    
    b.center.x() = 2.0f + 1e-6f;
    EXPECT_FALSE(intersectSphereSphere(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(2, 0, 0), Eigen::Vector3f(3, 0, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
    
    b.p0.x() = 2.0f + 1e-6f;
    b.p1.x() = 3.0f + 1e-6f;
    EXPECT_FALSE(intersectCapsuleCapsule(a, b));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(-2, 1, 0), Eigen::Vector3f(2, 1, 0)};
    Sphere sph{Eigen::Vector3f(0, 0, 0), 1.0f};
    EXPECT_TRUE(intersectSegmentSphere(seg, sph));
    
    seg.start.y() = 1.0f + 1e-6f;
    seg.end.y() = 1.0f + 1e-6f;
    EXPECT_FALSE(intersectSegmentSphere(seg, sph));
  }
}

TEST_F(IntersectionTest, SegmentCapsuleIntersection) 
{
  {
    LineSegment seg{Eigen::Vector3f(-2, 0, 0), Eigen::Vector3f(2, 0, 0)};
    Capsule cap{Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    EXPECT_TRUE(intersectSegmentCapsule(seg, cap));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(0, -2, 0), Eigen::Vector3f(0, 2, 0)};
    Capsule cap{Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    EXPECT_TRUE(intersectSegmentCapsule(seg, cap));
  }
  
  {
    LineSegment seg{Eigen::Vector3f(2, 2, 0), Eigen::Vector3f(3, 3, 0)};
    Capsule cap{Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    EXPECT_FALSE(intersectSegmentCapsule(seg, cap));
  }
}

TEST_F(IntersectionTest, BatchSphereSphereIntersection) 
{
  constexpr size_t count = 1024;
  std::vector<Sphere> spheres_a(count);
  std::vector<Sphere> spheres_b(count);
  std::vector<uint8_t> results(count);
  std::vector<uint8_t> results_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    spheres_a[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_a[i].radius = radius_dist_(rng_);
    
    spheres_b[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_b[i].radius = radius_dist_(rng_);
    
    results_ref[i] = intersectSphereSphere(spheres_a[i], spheres_b[i]) ? 1 : 0;
  }
  
  batchIntersectSphereSphere(spheres_a.data(), spheres_b.data(), count, 
                             reinterpret_cast<bool*>(results.data()));
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(results[i], results_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(IntersectionTest, BatchCapsuleCapsuleIntersection) 
{
  constexpr size_t count = 256;
  std::vector<Capsule> capsules_a(count);
  std::vector<Capsule> capsules_b(count);
  std::vector<uint8_t> results(count);
  std::vector<uint8_t> results_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    capsules_a[i].p0 = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    Eigen::Vector3f dir_a(unit_dist_(rng_), unit_dist_(rng_), unit_dist_(rng_));
    if (dir_a.norm() > 0.1f) dir_a.normalize();
    else dir_a = Eigen::Vector3f(1, 0, 0);
    capsules_a[i].p1 = capsules_a[i].p0 + dir_a * radius_dist_(rng_);
    capsules_a[i].radius = radius_dist_(rng_);
    
    capsules_b[i].p0 = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    Eigen::Vector3f dir_b(unit_dist_(rng_), unit_dist_(rng_), unit_dist_(rng_));
    if (dir_b.norm() > 0.1f) dir_b.normalize();
    else dir_b = Eigen::Vector3f(0, 1, 0);
    capsules_b[i].p1 = capsules_b[i].p0 + dir_b * radius_dist_(rng_);
    capsules_b[i].radius = radius_dist_(rng_);
    
    results_ref[i] = intersectCapsuleCapsule(capsules_a[i], capsules_b[i]) ? 1 : 0;
  }
  
  batchIntersectCapsuleCapsule(capsules_a.data(), capsules_b.data(), count, 
                               reinterpret_cast<bool*>(results.data()));
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(results[i], results_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(IntersectionTest, NumericalStability) 
{
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1e-6f};
    Sphere b{Eigen::Vector3f(1e-6f, 0, 0), 1e-6f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
  
  {
    Sphere a{Eigen::Vector3f(0, 0, 0), 1e6f};
    Sphere b{Eigen::Vector3f(1e6f, 0, 0), 1e6f};
    EXPECT_TRUE(intersectSphereSphere(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), 0.5f};
    Capsule b{Eigen::Vector3f(0, 1e-6f, 0), Eigen::Vector3f(1, 1e-6f, 0), 0.5f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
  
  {
    Capsule a{Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0), 1e-6f};
    Capsule b{Eigen::Vector3f(5, -1, 0), Eigen::Vector3f(5, 1, 0), 1.0f};
    EXPECT_TRUE(intersectCapsuleCapsule(a, b));
  }
}

TEST_F(IntersectionTest, UnalignedBatchOperations) 
{
  constexpr size_t count = 100;
  constexpr size_t offset = 1;
  
  std::vector<uint8_t> buffer_a(sizeof(Sphere) * (count + 1) + offset);
  std::vector<uint8_t> buffer_b(sizeof(Sphere) * (count + 1) + offset);
  std::vector<uint8_t> buffer_results(sizeof(bool) * (count + 1) + offset);
  
  Sphere* spheres_a = reinterpret_cast<Sphere*>(buffer_a.data() + offset);
  Sphere* spheres_b = reinterpret_cast<Sphere*>(buffer_b.data() + offset);
  bool* results = reinterpret_cast<bool*>(buffer_results.data() + offset);
  
  std::vector<uint8_t> results_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    spheres_a[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_a[i].radius = radius_dist_(rng_);
    
    spheres_b[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_b[i].radius = radius_dist_(rng_);
    
    results_ref[i] = intersectSphereSphere(spheres_a[i], spheres_b[i]) ? 1 : 0;
  }
  
  batchIntersectSphereSphere(spheres_a, spheres_b, count, results);
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(results[i] ? 1 : 0, results_ref[i]) 
      << "Unaligned batch operation failed at index " << i;
  }
}

TEST_F(IntersectionTest, ArrayEndMasking) 
{
  std::vector<size_t> test_sizes = {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65};
  
  for (size_t count : test_sizes) 
  {
    std::vector<Sphere> spheres_a(count);
    std::vector<Sphere> spheres_b(count);
    std::vector<uint8_t> results(count);
    std::vector<uint8_t> results_ref(count);
    
    for (size_t i = 0; i < count; ++i) 
    {
      spheres_a[i].center = Eigen::Vector3f(i, 0, 0);
      spheres_a[i].radius = 1.0f;
      
      spheres_b[i].center = Eigen::Vector3f(i + 0.5f, 0, 0);
      spheres_b[i].radius = 1.0f;
      
      results_ref[i] = intersectSphereSphere(spheres_a[i], spheres_b[i]) ? 1 : 0;
    }
    
    batchIntersectSphereSphere(spheres_a.data(), spheres_b.data(), count, 
                               reinterpret_cast<bool*>(results.data()));
    
    for (size_t i = 0; i < count; ++i) 
    {
      EXPECT_EQ(results[i], results_ref[i]) 
        << "Array end masking failed for count=" << count << " at index " << i;
    }
  }
}

TEST_F(IntersectionTest, BatchPerformance) 
{
  constexpr size_t large_count = 100000;
  std::vector<Sphere> spheres_a(large_count);
  std::vector<Sphere> spheres_b(large_count);
  std::vector<uint8_t> results(large_count);
  
  for (size_t i = 0; i < large_count; ++i) 
  {
    spheres_a[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_a[i].radius = radius_dist_(rng_);
    
    spheres_b[i].center = Eigen::Vector3f(pos_dist_(rng_), pos_dist_(rng_), pos_dist_(rng_));
    spheres_b[i].radius = radius_dist_(rng_);
  }
  
  ASSERT_NO_THROW(batchIntersectSphereSphere(spheres_a.data(), spheres_b.data(), 
                                              large_count, reinterpret_cast<bool*>(results.data())));
  
  for (size_t i = 0; i < 100; ++i) 
  {
    bool ref_result = intersectSphereSphere(spheres_a[i], spheres_b[i]);
    EXPECT_EQ(results[i] != 0, ref_result) << "Mismatch at index " << i;
  }
}

}
}
