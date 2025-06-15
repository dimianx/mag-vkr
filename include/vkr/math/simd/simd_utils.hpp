#ifndef VKR_MATH_SIMD_SIMD_UTILS_HPP_
#define VKR_MATH_SIMD_SIMD_UTILS_HPP_

#include <xsimd/xsimd.hpp>
#include <cstddef>
#include <memory>
#include <spdlog/spdlog.h>

namespace vkr
{
namespace math
{
namespace simd
{

namespace xs = xsimd;

using arch_type = xs::default_arch;
using batch_type = xs::batch<float, arch_type>;
using batch_bool_type = xs::batch_bool<float, arch_type>;

constexpr size_t SIMD_WIDTH = batch_type::size;
constexpr size_t SIMD_ALIGNMENT = arch_type::alignment();

inline void logSIMDInfo()
{
  spdlog::info("SIMD: Using {} with vector size {}",
               xs::default_arch::name(), SIMD_WIDTH);
}

inline bool isAligned(const void* ptr)
{
  return reinterpret_cast<uintptr_t>(ptr) % SIMD_ALIGNMENT == 0;
}

inline size_t alignUp(size_t size)
{
  return (size + SIMD_WIDTH - 1) & ~(SIMD_WIDTH - 1);
}

template<typename T>
struct AlignedDeleter
{
  void operator()(T* ptr) const
  {
    xs::aligned_free(ptr);
  }
};

template<typename T>
using AlignedPtr = std::unique_ptr<T[], AlignedDeleter<T>>;

template<typename T>
AlignedPtr<T> allocateAligned(size_t count)
{
  return AlignedPtr<T>(static_cast<T*>(
      xs::aligned_malloc(count * sizeof(T), SIMD_ALIGNMENT)));
}

template<typename Func>
inline void simdLoop(size_t count, Func&& func)
{
  const size_t simd_count = count & ~(SIMD_WIDTH - 1);
  
  for (size_t i = 0; i < simd_count; i += SIMD_WIDTH)
  {
    func(i, true);
  }
  
  for (size_t i = simd_count; i < count; ++i)
  {
    func(i, false);
  }
}

inline batch_type reciprocal(const batch_type& x)
{
  return 1.0f / x;
}

inline batch_type rsqrt(const batch_type& x)
{
  return xs::rsqrt(x);
}

inline batch_type clamp(const batch_type& x, float min_val, float max_val)
{
  return xs::min(xs::max(x, batch_type(min_val)), batch_type(max_val));
}

inline float horizontalSum(const batch_type& x)
{
  return xs::reduce_add(x);
}

inline float horizontalMin(const batch_type& x)
{
  return xs::reduce_min(x);
}

inline float horizontalMax(const batch_type& x)
{
  return xs::reduce_max(x);
}

}
}
}

#endif