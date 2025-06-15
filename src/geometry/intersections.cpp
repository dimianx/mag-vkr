#include "vkr/geometry/intersections.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <xsimd/xsimd.hpp>

namespace vkr
{
namespace geometry
{

void batchIntersectSphereSphere(const Sphere* a, const Sphere* b, 
                                size_t count, bool* results)
{
  using namespace math::simd;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width)
  {
    alignas(SIMD_ALIGNMENT) float ax[simd_width], ay[simd_width], az[simd_width];
    alignas(SIMD_ALIGNMENT) float bx[simd_width], by[simd_width], bz[simd_width];
    alignas(SIMD_ALIGNMENT) float ar[simd_width], br[simd_width];
    
    for (size_t j = 0; j < simd_width; ++j)
    {
      ax[j] = a[i + j].center.x();
      ay[j] = a[i + j].center.y();
      az[j] = a[i + j].center.z();
      ar[j] = a[i + j].radius;
      
      bx[j] = b[i + j].center.x();
      by[j] = b[i + j].center.y();
      bz[j] = b[i + j].center.z();
      br[j] = b[i + j].radius;
    }
    
    auto ax_v = batch_type::load_aligned(ax);
    auto ay_v = batch_type::load_aligned(ay);
    auto az_v = batch_type::load_aligned(az);
    auto ar_v = batch_type::load_aligned(ar);
    
    auto bx_v = batch_type::load_aligned(bx);
    auto by_v = batch_type::load_aligned(by);
    auto bz_v = batch_type::load_aligned(bz);
    auto br_v = batch_type::load_aligned(br);
    
    auto dx = ax_v - bx_v;
    auto dy = ay_v - by_v;
    auto dz = az_v - bz_v;
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    
    auto radius_sum = ar_v + br_v;
    auto radius_sum_sq = radius_sum * radius_sum;
    
    auto mask = dist_sq <= radius_sum_sq;
    
    for (size_t j = 0; j < simd_width; ++j)
    {
      results[i + j] = mask.get(j);
    }
  }
  
  for (size_t i = simd_count; i < count; ++i)
  {
    results[i] = intersectSphereSphere(a[i], b[i]);
  }
}

void batchIntersectCapsuleCapsule(const Capsule* a, const Capsule* b, 
                                  size_t count, bool* results)
{
  using namespace math::simd;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width)
  {
    alignas(SIMD_ALIGNMENT) float p0x_a[simd_width], p0y_a[simd_width], p0z_a[simd_width];
    alignas(SIMD_ALIGNMENT) float p1x_a[simd_width], p1y_a[simd_width], p1z_a[simd_width];
    alignas(SIMD_ALIGNMENT) float p0x_b[simd_width], p0y_b[simd_width], p0z_b[simd_width];
    alignas(SIMD_ALIGNMENT) float p1x_b[simd_width], p1y_b[simd_width], p1z_b[simd_width];
    alignas(SIMD_ALIGNMENT) float radius_a[simd_width], radius_b[simd_width];
    
    for (size_t j = 0; j < simd_width; ++j)
    {
      p0x_a[j] = a[i + j].p0.x();
      p0y_a[j] = a[i + j].p0.y();
      p0z_a[j] = a[i + j].p0.z();
      p1x_a[j] = a[i + j].p1.x();
      p1y_a[j] = a[i + j].p1.y();
      p1z_a[j] = a[i + j].p1.z();
      radius_a[j] = a[i + j].radius;
      
      p0x_b[j] = b[i + j].p0.x();
      p0y_b[j] = b[i + j].p0.y();
      p0z_b[j] = b[i + j].p0.z();
      p1x_b[j] = b[i + j].p1.x();
      p1y_b[j] = b[i + j].p1.y();
      p1z_b[j] = b[i + j].p1.z();
      radius_b[j] = b[i + j].radius;
    }
    
    auto p0x_a_v = batch_type::load_aligned(p0x_a);
    auto p0y_a_v = batch_type::load_aligned(p0y_a);
    auto p0z_a_v = batch_type::load_aligned(p0z_a);
    auto p1x_a_v = batch_type::load_aligned(p1x_a);
    auto p1y_a_v = batch_type::load_aligned(p1y_a);
    auto p1z_a_v = batch_type::load_aligned(p1z_a);
    auto radius_a_v = batch_type::load_aligned(radius_a);
    
    auto p0x_b_v = batch_type::load_aligned(p0x_b);
    auto p0y_b_v = batch_type::load_aligned(p0y_b);
    auto p0z_b_v = batch_type::load_aligned(p0z_b);
    auto p1x_b_v = batch_type::load_aligned(p1x_b);
    auto p1y_b_v = batch_type::load_aligned(p1y_b);
    auto p1z_b_v = batch_type::load_aligned(p1z_b);
    auto radius_b_v = batch_type::load_aligned(radius_b);
    
    auto ux = p1x_a_v - p0x_a_v;
    auto uy = p1y_a_v - p0y_a_v;
    auto uz = p1z_a_v - p0z_a_v;
    
    auto vx = p1x_b_v - p0x_b_v;
    auto vy = p1y_b_v - p0y_b_v;
    auto vz = p1z_b_v - p0z_b_v;
    
    auto wx = p0x_a_v - p0x_b_v;
    auto wy = p0y_a_v - p0y_b_v;
    auto wz = p0z_a_v - p0z_b_v;
    
    auto a_param = ux * ux + uy * uy + uz * uz;
    auto b_param = ux * vx + uy * vy + uz * vz;
    auto c_param = vx * vx + vy * vy + vz * vz;
    auto d_param = ux * wx + uy * wy + uz * wz;
    auto e_param = vx * wx + vy * wy + vz * wz;
    
    auto denom = a_param * c_param - b_param * b_param;
    
    const auto eps = batch_type(GEOM_EPS);
    auto parallel_mask = xsimd::abs(denom) < eps;
    
    auto s = (b_param * e_param - c_param * d_param) / denom;
    auto t = (a_param * e_param - b_param * d_param) / denom;
    
    s = xsimd::select(parallel_mask, batch_type(0.0f), s);
    t = xsimd::select(parallel_mask, e_param / c_param, t);
    
    const auto zero = batch_type(0.0f);
    const auto one = batch_type(1.0f);
    
    s = xsimd::max(zero, xsimd::min(one, s));
    t = xsimd::max(zero, xsimd::min(one, t));
    
    auto closest_ax = p0x_a_v + s * ux;
    auto closest_ay = p0y_a_v + s * uy;
    auto closest_az = p0z_a_v + s * uz;
    
    auto closest_bx = p0x_b_v + t * vx;
    auto closest_by = p0y_b_v + t * vy;
    auto closest_bz = p0z_b_v + t * vz;
    
    auto dx = closest_ax - closest_bx;
    auto dy = closest_ay - closest_by;
    auto dz = closest_az - closest_bz;
    auto dist_sq = dx * dx + dy * dy + dz * dz;
    
    auto radius_sum = radius_a_v + radius_b_v;
    auto radius_sum_sq = radius_sum * radius_sum;
    
    auto mask = dist_sq <= radius_sum_sq;
    
    alignas(SIMD_ALIGNMENT) float dist_sq_arr[simd_width];
    alignas(SIMD_ALIGNMENT) float radius_sum_sq_arr[simd_width];
    dist_sq.store_aligned(dist_sq_arr);
    radius_sum_sq.store_aligned(radius_sum_sq_arr);
    
    for (size_t j = 0; j < simd_width; ++j)
    {
      bool need_scalar = false;
      
      float diff = std::abs(dist_sq_arr[j] - radius_sum_sq_arr[j]);
      if (diff < 1e-4f)
        need_scalar = true;
      
      float len_a_sq = (a[i + j].p1 - a[i + j].p0).squaredNorm();
      float len_b_sq = (b[i + j].p1 - b[i + j].p0).squaredNorm();
      if (len_a_sq < GEOM_EPS || len_b_sq < GEOM_EPS)
        need_scalar = true;
      
      if (need_scalar)
      {
        results[i + j] = intersectCapsuleCapsule(a[i + j], b[i + j]);
      }
      else
      {
        results[i + j] = mask.get(j);
      }
    }
  }
  
  for (size_t i = simd_count; i < count; ++i)
  {
    results[i] = intersectCapsuleCapsule(a[i], b[i]);
  }
}

}
}