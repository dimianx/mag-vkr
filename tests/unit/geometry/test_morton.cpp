#include <gtest/gtest.h>
#include "vkr/geometry/morton.hpp"
#include <random>
#include <vector>
#include <chrono>

namespace vkr 
{
namespace geometry 
{

class MortonTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    rng_.seed(42);
  }
  
  std::mt19937 rng_;
  std::uniform_int_distribution<uint32_t> dist_3d_{0, (1 << 10) - 1};
  std::uniform_int_distribution<uint32_t> dist_3d_21_{0, (1 << 21) - 1};
  std::uniform_int_distribution<uint16_t> dist_2d_16_{0, 65535};
  std::uniform_int_distribution<uint32_t> dist_2d_32_{0, 4294967295u};
};

TEST_F(MortonTest, Morton3D_10bit_EncodeDecode) 
{
  {
    uint32_t x = 0, y = 0, z = 0;
    uint64_t morton = mortonEncode3D(x, y, z);
    EXPECT_EQ(morton, 0u);
    
    uint32_t dx, dy, dz;
    mortonDecode3D(morton, dx, dy, dz);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
    EXPECT_EQ(dz, z);
  }
  
  {
    uint32_t x = 1023, y = 1023, z = 1023;
    uint64_t morton = mortonEncode3D(x, y, z);
    
    uint32_t dx, dy, dz;
    mortonDecode3D(morton, dx, dy, dz);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
    EXPECT_EQ(dz, z);
  }
  
  for (int i = 0; i < 1000; ++i) 
  {
    uint32_t x = dist_3d_(rng_);
    uint32_t y = dist_3d_(rng_);
    uint32_t z = dist_3d_(rng_);
    
    uint64_t morton = mortonEncode3D(x, y, z);
    
    uint32_t dx, dy, dz;
    mortonDecode3D(morton, dx, dy, dz);
    
    EXPECT_EQ(dx, x) << "Failed at iteration " << i;
    EXPECT_EQ(dy, y) << "Failed at iteration " << i;
    EXPECT_EQ(dz, z) << "Failed at iteration " << i;
  }
}

TEST_F(MortonTest, Morton3D_21bit_EncodeDecode) 
{
  {
    uint32_t x = 0, y = 0, z = 0;
    uint64_t morton = morton3d::encode(x, y, z);
    EXPECT_EQ(morton, 0u);
    
    uint32_t dx, dy, dz;
    morton3d::decode(morton, dx, dy, dz);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
    EXPECT_EQ(dz, z);
  }
  
  {
    uint32_t x = (1 << 21) - 1;
    uint32_t y = (1 << 21) - 1;
    uint32_t z = (1 << 21) - 1;
    uint64_t morton = morton3d::encode(x, y, z);
    
    uint32_t dx, dy, dz;
    morton3d::decode(morton, dx, dy, dz);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
    EXPECT_EQ(dz, z);
  }
  
  for (int i = 0; i < 1000; ++i) 
  {
    uint32_t x = dist_3d_21_(rng_);
    uint32_t y = dist_3d_21_(rng_);
    uint32_t z = dist_3d_21_(rng_);
    
    uint64_t morton = morton3d::encode(x, y, z);
    
    uint32_t dx, dy, dz;
    morton3d::decode(morton, dx, dy, dz);
    
    EXPECT_EQ(dx, x) << "Failed at iteration " << i;
    EXPECT_EQ(dy, y) << "Failed at iteration " << i;
    EXPECT_EQ(dz, z) << "Failed at iteration " << i;
  }
}

TEST_F(MortonTest, Morton2D_16bit_EncodeDecode) 
{
  {
    uint16_t x = 0, y = 0;
    uint32_t morton = mortonEncode2D(x, y);
    EXPECT_EQ(morton, 0u);
    
    uint16_t dx, dy;
    mortonDecode2D(morton, dx, dy);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
  }
  
  for (int i = 0; i < 1000; ++i) 
  {
    uint16_t x = dist_2d_16_(rng_);
    uint16_t y = dist_2d_16_(rng_);
    
    uint32_t morton = mortonEncode2D(x, y);
    
    uint16_t dx, dy;
    mortonDecode2D(morton, dx, dy);
    
    EXPECT_EQ(dx, x) << "Failed at iteration " << i;
    EXPECT_EQ(dy, y) << "Failed at iteration " << i;
  }
}

TEST_F(MortonTest, Morton2D_32bit_EncodeDecode) 
{
  {
    uint32_t x = 0, y = 0;
    uint64_t morton = morton2d::encode(x, y);
    EXPECT_EQ(morton, 0u);
    
    uint32_t dx, dy;
    morton2d::decode(morton, dx, dy);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
  }
  
  {
    uint32_t x = 4294967295u;
    uint32_t y = 4294967295u;
    uint64_t morton = morton2d::encode(x, y);
    
    uint32_t dx, dy;
    morton2d::decode(morton, dx, dy);
    EXPECT_EQ(dx, x);
    EXPECT_EQ(dy, y);
  }
}

TEST_F(MortonTest, BatchMorton3D) 
{
  constexpr size_t count = 1024;
  std::vector<uint32_t> x(count), y(count), z(count);
  std::vector<uint64_t> morton(count);
  std::vector<uint64_t> morton_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    x[i] = dist_3d_(rng_);
    y[i] = dist_3d_(rng_);
    z[i] = dist_3d_(rng_);
    morton_ref[i] = mortonEncode3D(x[i], y[i], z[i]);
  }
  
  batchMortonEncode3D(x.data(), y.data(), z.data(), count, morton.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(morton[i], morton_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(MortonTest, BatchMorton2D) 
{
  constexpr size_t count = 1024;
  std::vector<uint16_t> x(count), y(count);
  std::vector<uint32_t> morton(count);
  std::vector<uint32_t> morton_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    x[i] = dist_2d_16_(rng_);
    y[i] = dist_2d_16_(rng_);
    morton_ref[i] = mortonEncode2D(x[i], y[i]);
  }
  
  batchMortonEncode2D(x.data(), y.data(), count, morton.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(morton[i], morton_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(MortonTest, BatchMorton3D21) 
{
  constexpr size_t count = 512;
  std::vector<uint32_t> x(count), y(count), z(count);
  std::vector<uint64_t> morton(count);
  std::vector<uint64_t> morton_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    x[i] = dist_3d_21_(rng_);
    y[i] = dist_3d_21_(rng_);
    z[i] = dist_3d_21_(rng_);
    morton_ref[i] = morton3d::encode(x[i], y[i], z[i]);
  }
  
  batchMortonEncode3D21(x.data(), y.data(), z.data(), count, morton.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(morton[i], morton_ref[i]) << "Mismatch at index " << i 
      << " for (" << x[i] << ", " << y[i] << ", " << z[i] << ")";
  }
}

TEST_F(MortonTest, BatchMorton2D32) 
{
  constexpr size_t count = 256;
  std::vector<uint32_t> x(count), y(count);
  std::vector<uint64_t> morton(count);
  std::vector<uint64_t> morton_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    x[i] = dist_2d_32_(rng_);
    y[i] = dist_2d_32_(rng_);
    morton_ref[i] = morton2d::encode(x[i], y[i]);
  }
  
  batchMortonEncode2D32(x.data(), y.data(), count, morton.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(morton[i], morton_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(MortonTest, MortonOrdering) 
{
  {
    uint16_t x1 = 0, y1 = 0;
    uint16_t x2 = 1, y2 = 0;
    uint16_t x3 = 0, y3 = 1;
    uint16_t x4 = 1, y4 = 1;
    
    uint32_t m1 = mortonEncode2D(x1, y1);
    uint32_t m2 = mortonEncode2D(x2, y2);
    uint32_t m3 = mortonEncode2D(x3, y3);
    uint32_t m4 = mortonEncode2D(x4, y4);
    
    EXPECT_LT(m1, m2);
    EXPECT_LT(m2, m3);
    EXPECT_LT(m3, m4);
  }
  
  {
    uint32_t coords[][3] = 
    {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };
    
    std::vector<uint64_t> morton_codes;
    for (auto& coord : coords) 
    {
      morton_codes.push_back(mortonEncode3D(coord[0], coord[1], coord[2]));
    }
    
    for (size_t i = 1; i < morton_codes.size(); ++i) 
    {
      EXPECT_LT(morton_codes[i-1], morton_codes[i]) 
        << "Ordering violated at index " << i;
    }
  }
}

TEST_F(MortonTest, NonAlignedSizes) 
{
  std::vector<size_t> test_sizes = {1, 3, 7, 13, 17, 31, 63, 127, 255, 513};
  
  for (size_t count : test_sizes) 
  {
    std::vector<uint32_t> x(count), y(count), z(count);
    std::vector<uint64_t> morton(count);
    std::vector<uint64_t> morton_ref(count);
    
    for (size_t i = 0; i < count; ++i) 
    {
      x[i] = dist_3d_(rng_);
      y[i] = dist_3d_(rng_);
      z[i] = dist_3d_(rng_);
      morton_ref[i] = mortonEncode3D(x[i], y[i], z[i]);
    }
    
    batchMortonEncode3D(x.data(), y.data(), z.data(), count, morton.data());
    
    for (size_t i = 0; i < count; ++i) 
    {
      EXPECT_EQ(morton[i], morton_ref[i]) 
        << "Failed for count=" << count << " at index " << i;
    }
  }
}

}
}
