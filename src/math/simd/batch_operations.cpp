#include "vkr/math/simd/batch_operations.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <xsimd/xsimd.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>

namespace vkr
{
namespace math
{
namespace simd
{

void batchInterpolate(const float* heights, size_t count, 
                     const float* u, const float* v,
                     float* output) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width) 
  {
    auto u_batch = batch_type::load_unaligned(&u[i]);
    auto v_batch = batch_type::load_unaligned(&v[i]);
    
    auto one = batch_type(1.0f);
    auto one_minus_u = one - u_batch;
    auto one_minus_v = one - v_batch;
    
    batch_type h00, h10, h01, h11;
    
    alignas(SIMD_ALIGNMENT) float h00_arr[simd_width];
    alignas(SIMD_ALIGNMENT) float h10_arr[simd_width];
    alignas(SIMD_ALIGNMENT) float h01_arr[simd_width];
    alignas(SIMD_ALIGNMENT) float h11_arr[simd_width];
    
    for (size_t j = 0; j < simd_width; ++j) 
    {
      size_t idx = (i + j) * 4;
      h00_arr[j] = heights[idx + 0];
      h10_arr[j] = heights[idx + 1];
      h01_arr[j] = heights[idx + 2];
      h11_arr[j] = heights[idx + 3];
    }
    
    h00 = batch_type::load_aligned(h00_arr);
    h10 = batch_type::load_aligned(h10_arr);
    h01 = batch_type::load_aligned(h01_arr);
    h11 = batch_type::load_aligned(h11_arr);
    
    auto result = one_minus_u * one_minus_v * h00 +
                  u_batch * one_minus_v * h10 +
                  one_minus_u * v_batch * h01 +
                  u_batch * v_batch * h11;
    
    result.store_unaligned(&output[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    size_t idx = i * 4;
    float h00 = heights[idx + 0];
    float h10 = heights[idx + 1];
    float h01 = heights[idx + 2];
    float h11 = heights[idx + 3];
    
    float u_val = u[i];
    float v_val = v[i];
    
    output[i] = (1.0f - u_val) * (1.0f - v_val) * h00 +
                u_val * (1.0f - v_val) * h10 +
                (1.0f - u_val) * v_val * h01 +
                u_val * v_val * h11;
  }
}

void batchDistanceCompute(const Eigen::Vector3f* points, size_t count,
                         const Eigen::Vector3f& query, 
                         float* distances) 
{
  Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(
      reinterpret_cast<const float*>(points), 3, count);
  
  Eigen::Map<Eigen::VectorXf>(distances, count) = 
      (points_mat.colwise() - query).colwise().norm();
}

void batchSquaredDistanceCompute(const Eigen::Vector3f* points, size_t count,
                                const Eigen::Vector3f& query,
                                float* squared_distances) 
{
  Eigen::Map<const Eigen::Matrix<float, 3, Eigen::Dynamic>> points_mat(
      reinterpret_cast<const float*>(points), 3, count);
  
  Eigen::Map<Eigen::VectorXf>(squared_distances, count) = 
      (points_mat.colwise() - query).colwise().squaredNorm();
}

void batchMin(const float* a, const float* b, size_t count, float* results) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width) 
  {
    auto a_batch = batch_type::load_unaligned(&a[i]);
    auto b_batch = batch_type::load_unaligned(&b[i]);
    
    auto nan_mask = xsimd::isnan(a_batch) | xsimd::isnan(b_batch);
    
    auto min_batch = xsimd::min(a_batch, b_batch);
    
    auto nan_val = batch_type(std::numeric_limits<float>::quiet_NaN());
    auto result = xsimd::select(nan_mask, nan_val, min_batch);
    
    result.store_unaligned(&results[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    if (std::isnan(a[i]) || std::isnan(b[i])) 
    {
      results[i] = std::numeric_limits<float>::quiet_NaN();
    } 
    else 
    {
      results[i] = std::min(a[i], b[i]);
    }
  }
}

void batchMax(const float* a, const float* b, size_t count, float* results) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width) 
  {
    auto a_batch = batch_type::load_unaligned(&a[i]);
    auto b_batch = batch_type::load_unaligned(&b[i]);
    
    auto nan_mask = xsimd::isnan(a_batch) | xsimd::isnan(b_batch);
    
    auto max_batch = xsimd::max(a_batch, b_batch);
    
    auto nan_val = batch_type(std::numeric_limits<float>::quiet_NaN());
    auto result = xsimd::select(nan_mask, nan_val, max_batch);
    
    result.store_unaligned(&results[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    if (std::isnan(a[i]) || std::isnan(b[i])) 
    {
      results[i] = std::numeric_limits<float>::quiet_NaN();
    } 
    else 
    {
      results[i] = std::max(a[i], b[i]);
    }
  }
}

void batchClamp(const float* values, size_t count,
               float min_value, float max_value,
               float* results) 
{
  Eigen::Map<Eigen::ArrayXf>(results, count) = 
      Eigen::Map<const Eigen::ArrayXf>(values, count)
          .max(min_value)
          .min(max_value);
}

void batchPolynomialEval(const float* coefficients, int degree,
                        const float* t_values, size_t count,
                        float* results) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const size_t simd_count = count & ~(simd_width - 1);
  
  for (size_t i = 0; i < simd_count; i += simd_width) 
  {
    auto t = batch_type::load_unaligned(&t_values[i]);
    auto result = batch_type(coefficients[degree]);
    
    for (int j = degree - 1; j >= 0; --j) 
    {
      result = result * t + batch_type(coefficients[j]);
    }
    
    result.store_unaligned(&results[i]);
  }
  
  for (size_t i = simd_count; i < count; ++i) 
  {
    float t = t_values[i];
    float result = coefficients[degree];
    
    for (int j = degree - 1; j >= 0; --j) 
    {
      result = result * t + coefficients[j];
    }
    
    results[i] = result;
  }
}

void batchHaarTransform1D(float* data, size_t batch_size, size_t length) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const float scale = 1.0f / std::sqrt(2.0f);
  const auto scale_vec = batch_type(scale);
  
  for (size_t len = length; len > 1; len /= 2) 
  {
    size_t half_len = len / 2;
    
    for (size_t b = 0; b < batch_size; ++b) 
    {
      float* batch_data = data + b * length;
      
      const size_t simd_pairs = half_len & ~(simd_width - 1);
      
      std::vector<float> temp(len);
      
      for (size_t i = 0; i < simd_pairs; i += simd_width) 
      {
        alignas(SIMD_ALIGNMENT) float even_arr[simd_width];
        alignas(SIMD_ALIGNMENT) float odd_arr[simd_width];
        
        for (size_t j = 0; j < simd_width; ++j)
        {
          even_arr[j] = batch_data[2 * (i + j)];
          odd_arr[j] = batch_data[2 * (i + j) + 1];
        }
        
        auto even = batch_type::load_aligned(even_arr);
        auto odd = batch_type::load_aligned(odd_arr);
        
        auto avg = (even + odd) * scale_vec;
        auto diff = (even - odd) * scale_vec;
        
        avg.store_unaligned(&temp[i]);
        diff.store_unaligned(&temp[half_len + i]);
      }
      
      for (size_t i = simd_pairs; i < half_len; ++i) 
      {
        float even = batch_data[2 * i];
        float odd = batch_data[2 * i + 1];
        temp[i] = (even + odd) * scale;
        temp[half_len + i] = (even - odd) * scale;
      }
      
      std::copy(temp.begin(), temp.begin() + len, batch_data);
    }
  }
}

void batchHaarInverse1D(float* data, size_t batch_size, size_t length) 
{
  using batch_type = batch_type;
  constexpr size_t simd_width = SIMD_WIDTH;
  
  const float scale = 1.0f / std::sqrt(2.0f);
  const auto scale_vec = batch_type(scale);
  
  for (size_t len = 2; len <= length; len *= 2) 
  {
    size_t half_len = len / 2;
    
    for (size_t b = 0; b < batch_size; ++b) 
    {
      float* batch_data = data + b * length;
      
      const size_t simd_pairs = half_len & ~(simd_width - 1);
      
      std::vector<float> temp(len);
      
      for (size_t i = 0; i < simd_pairs; i += simd_width) 
      {
        auto avg = batch_type::load_unaligned(&batch_data[i]);
        auto diff = batch_type::load_unaligned(&batch_data[half_len + i]);
        
        auto even = (avg + diff) * scale_vec;
        auto odd = (avg - diff) * scale_vec;
        
        alignas(SIMD_ALIGNMENT) float even_arr[simd_width];
        alignas(SIMD_ALIGNMENT) float odd_arr[simd_width];
        even.store_aligned(even_arr);
        odd.store_aligned(odd_arr);
        
        for (size_t j = 0; j < simd_width; ++j) 
        {
          temp[2 * (i + j)] = even_arr[j];
          temp[2 * (i + j) + 1] = odd_arr[j];
        }
      }
      
      for (size_t i = simd_pairs; i < half_len; ++i) 
      {
        float avg = batch_data[i];
        float diff = batch_data[half_len + i];
        temp[2 * i] = (avg + diff) * scale;
        temp[2 * i + 1] = (avg - diff) * scale;
      }
      
      std::copy(temp.begin(), temp.begin() + len, batch_data);
    }
  }
}

}
}
}