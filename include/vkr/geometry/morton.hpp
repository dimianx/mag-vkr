#ifndef VKR_GEOMETRY_MORTON_HPP_
#define VKR_GEOMETRY_MORTON_HPP_

#include <cstdint>
#include <cstddef>

namespace vkr
{
namespace geometry
{

namespace morton3d
{

inline uint64_t expandBits(uint32_t v) 
{
  uint64_t x = v & 0x1fffff;
  x = (x | (x << 32)) & 0x1f00000000ffffULL;
  x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
  x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
  x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
  x = (x | (x << 2))  & 0x1249249249249249ULL;
  return x;
}

inline uint32_t compactBits(uint64_t v) 
{
  v = v & 0x1249249249249249ULL;
  v = (v | (v >> 2))  & 0x10c30c30c30c30c3ULL;
  v = (v | (v >> 4))  & 0x100f00f00f00f00fULL;
  v = (v | (v >> 8))  & 0x1f0000ff0000ffULL;
  v = (v | (v >> 16)) & 0x1f00000000ffffULL;
  v = (v | (v >> 32)) & 0x1fffffULL;
  return static_cast<uint32_t>(v);
}

inline uint64_t encode(uint32_t x, uint32_t y, uint32_t z) 
{
  return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

inline void decode(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) 
{
  x = compactBits(morton);
  y = compactBits(morton >> 1);
  z = compactBits(morton >> 2);
}

}

namespace morton2d
{

inline uint64_t expandBits(uint32_t v) 
{
  uint64_t x = v;
  x = (x | (x << 16)) & 0x0000ffff0000ffffULL;
  x = (x | (x << 8))  & 0x00ff00ff00ff00ffULL;
  x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0fULL;
  x = (x | (x << 2))  & 0x3333333333333333ULL;
  x = (x | (x << 1))  & 0x5555555555555555ULL;
  return x;
}

inline uint32_t compactBits(uint64_t v) 
{
  v = v & 0x5555555555555555ULL;
  v = (v | (v >> 1))  & 0x3333333333333333ULL;
  v = (v | (v >> 2))  & 0x0f0f0f0f0f0f0f0fULL;
  v = (v | (v >> 4))  & 0x00ff00ff00ff00ffULL;
  v = (v | (v >> 8))  & 0x0000ffff0000ffffULL;
  v = (v | (v >> 16)) & 0x00000000ffffffffULL;
  return static_cast<uint32_t>(v);
}

inline uint64_t encode(uint32_t x, uint32_t y) 
{
  return (expandBits(y) << 1) | expandBits(x);
}

inline void decode(uint64_t morton, uint32_t& x, uint32_t& y) 
{
  x = compactBits(morton);
  y = compactBits(morton >> 1);
}

}

inline uint64_t mortonEncode3D(uint32_t x, uint32_t y, uint32_t z) 
{
  x &= 0x3ff;
  y &= 0x3ff;
  z &= 0x3ff;
  
  uint64_t answer = 0;
  for (uint64_t i = 0; i < 10; ++i) 
  {
    answer |= ((x & (1 << i)) << (2 * i)) | 
              ((y & (1 << i)) << (2 * i + 1)) | 
              ((z & (1 << i)) << (2 * i + 2));
  }
  return answer;
}

inline void mortonDecode3D(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) 
{
  x = y = z = 0;
  for (uint64_t i = 0; i < 10; ++i) 
  {
    x |= ((morton & (1ULL << (3 * i))) >> (2 * i));
    y |= ((morton & (1ULL << (3 * i + 1))) >> (2 * i + 1));
    z |= ((morton & (1ULL << (3 * i + 2))) >> (2 * i + 2));
  }
}

inline uint32_t mortonEncode2D(uint16_t x, uint16_t y) 
{
  uint32_t answer = 0;
  for (uint32_t i = 0; i < 16; ++i) 
  {
    answer |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
  }
  return answer;
}

inline void mortonDecode2D(uint32_t morton, uint16_t& x, uint16_t& y) 
{
  x = y = 0;
  for (uint32_t i = 0; i < 16; ++i) 
  {
    x |= ((morton & (1 << (2 * i))) >> i);
    y |= ((morton & (1 << (2 * i + 1))) >> (i + 1));
  }
}

void batchMortonEncode3D(const uint32_t* x, const uint32_t* y, const uint32_t* z, 
                         size_t count, uint64_t* morton);

void batchMortonEncode2D(const uint16_t* x, const uint16_t* y, 
                         size_t count, uint32_t* morton);

void batchMortonEncode3D21(const uint32_t* x, const uint32_t* y, const uint32_t* z,
                          size_t count, uint64_t* morton);

void batchMortonEncode2D32(const uint32_t* x, const uint32_t* y,
                          size_t count, uint64_t* morton);

}
}

#endif