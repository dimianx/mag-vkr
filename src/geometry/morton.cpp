#include "vkr/geometry/morton.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <xsimd/xsimd.hpp>
#include <array>
#include <utility>

namespace vkr
{
namespace geometry
{

namespace
{

template<typename Batch>
inline Batch expandBits2D_SIMD(const Batch& v) 
{
  Batch x = v;
  x = (x | (x << 16)) & Batch(0x0000ffff0000ffffULL);
  x = (x | (x << 8))  & Batch(0x00ff00ff00ff00ffULL);
  x = (x | (x << 4))  & Batch(0x0f0f0f0f0f0f0f0fULL);
  x = (x | (x << 2))  & Batch(0x3333333333333333ULL);
  x = (x | (x << 1))  & Batch(0x5555555555555555ULL);
  return x;
}

template<typename Batch>
inline Batch expandBits3D_10bit_SIMD(const Batch& v) 
{
  Batch x = v & Batch(0x3ff);
  
  x = (x | (x << 16)) & Batch(0x030000ffULL);
  x = (x | (x << 8))  & Batch(0x0300f00fULL);
  x = (x | (x << 4))  & Batch(0x030c30c3ULL);
  x = (x | (x << 2))  & Batch(0x09249249ULL);
  
  return x;
}

constexpr uint64_t compute_morton_lut_7bit(uint32_t val) 
{
  uint64_t x = val & 0x7f;
  
  x = (x | (x << 16)) & 0x00000003000000FFULL;
  x = (x | (x << 8))  & 0x000300F00F00F00FULL;
  x = (x | (x << 4))  & 0x030C30C30C30C30C3ULL;
  x = (x | (x << 2))  & 0x09249249249249249ULL;
  
  return x;
}

template<size_t... Is>
constexpr auto make_morton_lut_7bit(std::index_sequence<Is...>) 
{
  return std::array<uint64_t, sizeof...(Is)>
  {
    compute_morton_lut_7bit(Is)...
  };
}

constexpr auto morton_lut_7bit = make_morton_lut_7bit(std::make_index_sequence<128>
{
});

constexpr uint64_t compute_morton_lut_8bit_2d(uint32_t val) 
{
  uint64_t x = val & 0xff;
  x = (x | (x << 8))  & 0x00ff00ffULL;
  x = (x | (x << 4))  & 0x0f0f0f0fULL;
  x = (x | (x << 2))  & 0x33333333ULL;
  x = (x | (x << 1))  & 0x55555555ULL;
  return x;
}

template<size_t... Is>
constexpr auto make_morton_lut_8bit_2d(std::index_sequence<Is...>) 
{
  return std::array<uint64_t, sizeof...(Is)>
  {
    compute_morton_lut_8bit_2d(Is)...
  };
}

constexpr auto morton_lut_8bit_2d = make_morton_lut_8bit_2d(std::make_index_sequence<256>
{
});

}

void batchMortonEncode3D(const uint32_t* x, const uint32_t* y, const uint32_t* z, 
                         size_t count, uint64_t* morton) 
{
  namespace simd = math::simd;
  using batch_u32 = xsimd::batch<uint32_t, simd::arch_type>;
  using batch_u64 = xsimd::batch<uint64_t, simd::arch_type>;
  
  constexpr size_t u32_width = batch_u32::size;
  constexpr size_t u64_width = batch_u64::size;
  constexpr size_t chunk_size = (u32_width > u64_width) ? u64_width : u32_width;
  
  const size_t simd_count = count & ~(chunk_size - 1);
  
  for (size_t i = 0; i < simd_count; i += chunk_size) 
  {
    batch_u64 batch_x_64, batch_y_64, batch_z_64;
    
    alignas(64) uint64_t x_arr[chunk_size];
    alignas(64) uint64_t y_arr[chunk_size];
    alignas(64) uint64_t z_arr[chunk_size];
    
    for (size_t j = 0; j < chunk_size; ++j) 
    {
      x_arr[j] = static_cast<uint64_t>(x[i + j] & 0x3ff);
      y_arr[j] = static_cast<uint64_t>(y[i + j] & 0x3ff);
      z_arr[j] = static_cast<uint64_t>(z[i + j] & 0x3ff);
    }
    
    batch_x_64 = batch_u64::load_aligned(x_arr);
    batch_y_64 = batch_u64::load_aligned(y_arr);
    batch_z_64 = batch_u64::load_aligned(z_arr);
    
    batch_x_64 = (batch_x_64 | (batch_x_64 << 16)) & batch_u64(0x030000ffULL);
    batch_y_64 = (batch_y_64 | (batch_y_64 << 16)) & batch_u64(0x030000ffULL);
    batch_z_64 = (batch_z_64 | (batch_z_64 << 16)) & batch_u64(0x030000ffULL);
    
    batch_x_64 = (batch_x_64 | (batch_x_64 << 8)) & batch_u64(0x0300f00fULL);
    batch_y_64 = (batch_y_64 | (batch_y_64 << 8)) & batch_u64(0x0300f00fULL);
    batch_z_64 = (batch_z_64 | (batch_z_64 << 8)) & batch_u64(0x0300f00fULL);
    
    batch_x_64 = (batch_x_64 | (batch_x_64 << 4)) & batch_u64(0x030c30c3ULL);
    batch_y_64 = (batch_y_64 | (batch_y_64 << 4)) & batch_u64(0x030c30c3ULL);
    batch_z_64 = (batch_z_64 | (batch_z_64 << 4)) & batch_u64(0x030c30c3ULL);
    
    batch_x_64 = (batch_x_64 | (batch_x_64 << 2)) & batch_u64(0x09249249ULL);
    batch_y_64 = (batch_y_64 | (batch_y_64 << 2)) & batch_u64(0x09249249ULL);
    batch_z_64 = (batch_z_64 | (batch_z_64 << 2)) & batch_u64(0x09249249ULL);
    
    auto result = batch_x_64 | (batch_y_64 << 1) | (batch_z_64 << 2);
    
    result.store_unaligned(&morton[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    morton[i] = mortonEncode3D(x[i], y[i], z[i]);
  }
}

void batchMortonEncode2D(const uint16_t* x, const uint16_t* y, 
                         size_t count, uint32_t* morton) 
{
  namespace simd = math::simd;
  using batch_u32 = xsimd::batch<uint32_t, simd::arch_type>;
  
  constexpr size_t simd_width = batch_u32::size;
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width) 
  {
    batch_u32 batch_x, batch_y;
    
    alignas(64) uint32_t x_arr[simd_width];
    alignas(64) uint32_t y_arr[simd_width];
    
    for (size_t j = 0; j < simd_width; ++j) 
    {
      x_arr[j] = static_cast<uint32_t>(x[i + j]);
      y_arr[j] = static_cast<uint32_t>(y[i + j]);
    }
    
    batch_x = batch_u32::load_aligned(x_arr);
    batch_y = batch_u32::load_aligned(y_arr);
    
    batch_x = (batch_x | (batch_x << 8)) & batch_u32(0x00ff00ffU);
    batch_y = (batch_y | (batch_y << 8)) & batch_u32(0x00ff00ffU);
    
    batch_x = (batch_x | (batch_x << 4)) & batch_u32(0x0f0f0f0fU);
    batch_y = (batch_y | (batch_y << 4)) & batch_u32(0x0f0f0f0fU);
    
    batch_x = (batch_x | (batch_x << 2)) & batch_u32(0x33333333U);
    batch_y = (batch_y | (batch_y << 2)) & batch_u32(0x33333333U);
    
    batch_x = (batch_x | (batch_x << 1)) & batch_u32(0x55555555U);
    batch_y = (batch_y | (batch_y << 1)) & batch_u32(0x55555555U);
    
    auto result = batch_x | (batch_y << 1);
    
    result.store_unaligned(&morton[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    morton[i] = mortonEncode2D(x[i], y[i]);
  }
}

void batchMortonEncode3D21(const uint32_t* x, const uint32_t* y, const uint32_t* z,
                          size_t count, uint64_t* morton) 
{
  namespace simd = math::simd;
  using batch_u32 = xsimd::batch<uint32_t, simd::arch_type>;
  using batch_u64 = xsimd::batch<uint64_t, simd::arch_type>;
  
  constexpr size_t u32_width = batch_u32::size;
  constexpr size_t u64_width = batch_u64::size;
  constexpr size_t chunk_size = (u32_width > u64_width) ? u64_width : u32_width;
  
  const size_t simd_count = count & ~(chunk_size - 1);
  
  for (size_t i = 0; i < simd_count; i += chunk_size) 
  {
    batch_u64 batch_x, batch_y, batch_z;
    
    alignas(64) uint64_t x_arr[chunk_size];
    alignas(64) uint64_t y_arr[chunk_size];
    alignas(64) uint64_t z_arr[chunk_size];
    
    for (size_t j = 0; j < chunk_size; ++j) 
    {
      x_arr[j] = static_cast<uint64_t>(x[i + j] & 0x1fffff);
      y_arr[j] = static_cast<uint64_t>(y[i + j] & 0x1fffff);
      z_arr[j] = static_cast<uint64_t>(z[i + j] & 0x1fffff);
    }
    
    batch_x = batch_u64::load_aligned(x_arr);
    batch_y = batch_u64::load_aligned(y_arr);
    batch_z = batch_u64::load_aligned(z_arr);
    
    auto x_low = batch_x & batch_u64(0x7f);
    auto x_mid = (batch_x >> 7) & batch_u64(0x7f);
    auto x_hi = (batch_x >> 14) & batch_u64(0x7f);
    
    auto y_low = batch_y & batch_u64(0x7f);
    auto y_mid = (batch_y >> 7) & batch_u64(0x7f);
    auto y_hi = (batch_y >> 14) & batch_u64(0x7f);
    
    auto z_low = batch_z & batch_u64(0x7f);
    auto z_mid = (batch_z >> 7) & batch_u64(0x7f);
    auto z_hi = (batch_z >> 14) & batch_u64(0x7f);
    
    batch_u64 x_expanded, y_expanded, z_expanded;
    
    alignas(64) uint64_t x_exp_arr[chunk_size];
    alignas(64) uint64_t y_exp_arr[chunk_size];
    alignas(64) uint64_t z_exp_arr[chunk_size];
    
    for (size_t j = 0; j < chunk_size; ++j) 
    {
      uint64_t x_lo_exp = morton_lut_7bit[static_cast<size_t>(x_low.get(j))];
      uint64_t x_mi_exp = morton_lut_7bit[static_cast<size_t>(x_mid.get(j))];
      uint64_t x_hi_exp = morton_lut_7bit[static_cast<size_t>(x_hi.get(j))];
      
      uint64_t y_lo_exp = morton_lut_7bit[static_cast<size_t>(y_low.get(j))];
      uint64_t y_mi_exp = morton_lut_7bit[static_cast<size_t>(y_mid.get(j))];
      uint64_t y_hi_exp = morton_lut_7bit[static_cast<size_t>(y_hi.get(j))];
      
      uint64_t z_lo_exp = morton_lut_7bit[static_cast<size_t>(z_low.get(j))];
      uint64_t z_mi_exp = morton_lut_7bit[static_cast<size_t>(z_mid.get(j))];
      uint64_t z_hi_exp = morton_lut_7bit[static_cast<size_t>(z_hi.get(j))];
      
      x_exp_arr[j] = x_lo_exp | (x_mi_exp << 21) | (x_hi_exp << 42);
      y_exp_arr[j] = y_lo_exp | (y_mi_exp << 21) | (y_hi_exp << 42);
      z_exp_arr[j] = z_lo_exp | (z_mi_exp << 21) | (z_hi_exp << 42);
    }
    
    x_expanded = batch_u64::load_aligned(x_exp_arr);
    y_expanded = batch_u64::load_aligned(y_exp_arr);
    z_expanded = batch_u64::load_aligned(z_exp_arr);
    
    auto result = x_expanded | (y_expanded << 1) | (z_expanded << 2);
    
    result.store_unaligned(&morton[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    morton[i] = morton3d::encode(x[i], y[i], z[i]);
  }
}

void batchMortonEncode2D32(const uint32_t* x, const uint32_t* y,
                          size_t count, uint64_t* morton) 
{
  namespace simd = math::simd;
  using batch_u64 = xsimd::batch<uint64_t, simd::arch_type>;
  
  constexpr size_t u64_width = batch_u64::size;
  const size_t simd_count = count & ~(u64_width - 1);
  
  for (size_t i = 0; i < simd_count; i += u64_width) 
  {
    batch_u64 batch_x, batch_y;
    
    alignas(64) uint64_t x_arr[u64_width];
    alignas(64) uint64_t y_arr[u64_width];
    
    for (size_t j = 0; j < u64_width; ++j) 
    {
      x_arr[j] = static_cast<uint64_t>(x[i + j]);
      y_arr[j] = static_cast<uint64_t>(y[i + j]);
    }
    
    batch_x = batch_u64::load_aligned(x_arr);
    batch_y = batch_u64::load_aligned(y_arr);
    
    auto x_b0 = batch_x & batch_u64(0xff);
    auto x_b1 = (batch_x >> 8) & batch_u64(0xff);
    auto x_b2 = (batch_x >> 16) & batch_u64(0xff);
    auto x_b3 = (batch_x >> 24) & batch_u64(0xff);
    
    auto y_b0 = batch_y & batch_u64(0xff);
    auto y_b1 = (batch_y >> 8) & batch_u64(0xff);
    auto y_b2 = (batch_y >> 16) & batch_u64(0xff);
    auto y_b3 = (batch_y >> 24) & batch_u64(0xff);
    
    batch_u64 x_expanded, y_expanded;
    
    alignas(64) uint64_t x_exp_arr[u64_width];
    alignas(64) uint64_t y_exp_arr[u64_width];
    
    for (size_t j = 0; j < u64_width; ++j) 
    {
      uint64_t x_b0_exp = morton_lut_8bit_2d[static_cast<size_t>(x_b0.get(j))];
      uint64_t x_b1_exp = morton_lut_8bit_2d[static_cast<size_t>(x_b1.get(j))];
      uint64_t x_b2_exp = morton_lut_8bit_2d[static_cast<size_t>(x_b2.get(j))];
      uint64_t x_b3_exp = morton_lut_8bit_2d[static_cast<size_t>(x_b3.get(j))];
      
      uint64_t y_b0_exp = morton_lut_8bit_2d[static_cast<size_t>(y_b0.get(j))];
      uint64_t y_b1_exp = morton_lut_8bit_2d[static_cast<size_t>(y_b1.get(j))];
      uint64_t y_b2_exp = morton_lut_8bit_2d[static_cast<size_t>(y_b2.get(j))];
      uint64_t y_b3_exp = morton_lut_8bit_2d[static_cast<size_t>(y_b3.get(j))];
      
      x_exp_arr[j] = x_b0_exp | (x_b1_exp << 16) | (x_b2_exp << 32) | (x_b3_exp << 48);
      y_exp_arr[j] = y_b0_exp | (y_b1_exp << 16) | (y_b2_exp << 32) | (y_b3_exp << 48);
    }
    
    x_expanded = batch_u64::load_aligned(x_exp_arr);
    y_expanded = batch_u64::load_aligned(y_exp_arr);
    
    auto result = x_expanded | (y_expanded << 1);
    
    result.store_unaligned(&morton[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    morton[i] = morton2d::encode(x[i], y[i]);
  }
}

}
}