#include "vkr/math/wavelets/haar_wavelet_2d.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include "vkr/math/simd/batch_operations.hpp"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

namespace vkr
{
namespace math
{
namespace wavelets
{

namespace
{

constexpr float SQRT2 = 1.41421356237f;
constexpr float INV_SQRT2 = 0.70710678118f;
constexpr double INV_SQRT2_DOUBLE = 0.7071067811865475244;
constexpr double SQRT2_DOUBLE = 1.4142135623730950488;

size_t getMaxLevels(size_t width, size_t height)
{
  return static_cast<size_t>(std::floor(std::log2(std::min(width, height))));
}

void forwardCoreDouble(double* data, size_t width, size_t height, size_t levels)
{
  size_t max_possible_levels = getMaxLevels(width, height);
  size_t actual_levels = std::min(levels, max_possible_levels);
  
  size_t current_width = width;
  size_t current_height = height;
  
  std::vector<double> temp;
  
  for (size_t level = 0; level < actual_levels; ++level)
  {
    if (current_width < 2 || current_height < 2)
    {
      break;
    }
    
    temp.resize(current_width);
    for (size_t y = 0; y < current_height; ++y)
    {
      double* row = &data[y * width];
      size_t half_width = current_width / 2;
      
      std::copy(row, row + current_width, temp.begin());
      
      for (size_t i = 0; i < half_width; ++i)
      {
        double a = temp[2 * i];
        double b = temp[2 * i + 1];
        row[i] = (a + b) * INV_SQRT2_DOUBLE;
        row[half_width + i] = (a - b) * INV_SQRT2_DOUBLE;
      }
    }
    
    temp.resize(current_height);
    for (size_t x = 0; x < current_width; ++x)
    {
      for (size_t y = 0; y < current_height; ++y)
      {
        temp[y] = data[y * width + x];
      }
      
      size_t half_height = current_height / 2;
      
      for (size_t i = 0; i < half_height; ++i)
      {
        double a = temp[2 * i];
        double b = temp[2 * i + 1];
        data[i * width + x] = (a + b) * INV_SQRT2_DOUBLE;
        data[(half_height + i) * width + x] = (a - b) * INV_SQRT2_DOUBLE;
      }
    }
    
    current_width /= 2;
    current_height /= 2;
  }
}

void inverseCoreDouble(double* data, size_t width, size_t height, size_t levels)
{
  size_t max_possible_levels = getMaxLevels(width, height);
  size_t actual_levels = std::min(levels, max_possible_levels);
  
  size_t current_width = width >> actual_levels;
  size_t current_height = height >> actual_levels;
  current_width = std::max(size_t(1), current_width);
  current_height = std::max(size_t(1), current_height);
  
  std::vector<double> temp;
  
  for (size_t level = actual_levels; level > 0; --level)
  {
    current_width = std::min(width, current_width * 2);
    current_height = std::min(height, current_height * 2);
    
    temp.resize(current_height);
    for (size_t x = 0; x < current_width; ++x)
    {
      for (size_t y = 0; y < current_height; ++y)
      {
        temp[y] = data[y * width + x];
      }
      
      size_t half_height = current_height / 2;
      
      for (size_t i = 0; i < half_height; ++i)
      {
        double avg = temp[i] * INV_SQRT2_DOUBLE;
        double diff = temp[half_height + i] * INV_SQRT2_DOUBLE;
        data[(2 * i) * width + x] = avg + diff;
        data[(2 * i + 1) * width + x] = avg - diff;
      }
    }
    
    temp.resize(current_width);
    for (size_t y = 0; y < current_height; ++y)
    {
      double* row = &data[y * width];
      
      std::copy(row, row + current_width, temp.begin());
      
      size_t half_width = current_width / 2;
      
      for (size_t i = 0; i < half_width; ++i)
      {
        double avg = temp[i] * INV_SQRT2_DOUBLE;
        double diff = temp[half_width + i] * INV_SQRT2_DOUBLE;
        row[2 * i] = avg + diff;
        row[2 * i + 1] = avg - diff;
      }
    }
  }
}

}

void HaarWavelet2D::forward(float* data, size_t width, size_t height, size_t levels)
{
  if (levels == 0 || width < 2 || height < 2)
  {
    return;
  }
  
  const size_t N = width * height;
  std::vector<double> buf(N);
  
  for (size_t i = 0; i < N; ++i)
  {
    buf[i] = static_cast<double>(data[i]);
  }
  
  forwardCoreDouble(buf.data(), width, height, levels);
  
  for (size_t i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(buf[i]);
  }
}

void HaarWavelet2D::inverse(float* data, size_t width, size_t height, size_t levels)
{
  if (levels == 0 || width < 2 || height < 2)
  {
    return;
  }
  
  const size_t N = width * height;
  std::vector<double> buf(N);
  
  for (size_t i = 0; i < N; ++i)
  {
    buf[i] = static_cast<double>(data[i]);
  }
  
  inverseCoreDouble(buf.data(), width, height, levels);
  
  for (size_t i = 0; i < N; ++i)
  {
    data[i] = static_cast<float>(buf[i]);
  }
}

void HaarWavelet2D::forwardQuantized(float* data, size_t width, size_t height, 
                                     size_t levels, float quantization_step)
{
  forward(data, width, height, levels);
  
  if (quantization_step <= 1e-9f)
  {
    return;
  }
  
  const double inv_step = 1.0 / static_cast<double>(quantization_step);
  
  const size_t N = width * height;
  for (size_t i = 0; i < N; ++i)
  {
    const double k_double = std::round(static_cast<double>(data[i]) * inv_step);
    
    data[i] = static_cast<float>(k_double / inv_step);
  }
}

float HaarWavelet2D::forwardQuantizedWithError(float* data, size_t width, size_t height,
                                               size_t levels, float quantization_step)
{
  std::vector<float> original(data, data + width * height);
  
  forwardQuantized(data, width, height, levels, quantization_step);
  
  std::vector<float> quantized(data, data + width * height);
  
  inverse(data, width, height, levels);
  
  float max_error = 0.0f;
  for (size_t i = 0; i < width * height; ++i)
  {
    float error = std::abs(original[i] - data[i]);
    max_error = std::max(max_error, error);
  }
  
  std::copy(quantized.begin(), quantized.end(), data);
  
  return max_error;
}

void HaarWavelet2D::getCoefficientsAtLevel(const float* data, size_t width, size_t height,
                                          size_t level, float* output) const
{
  if (level == 0)
  {
    std::copy(data, data + width * height, output);
    return;
  }
  
  size_t max_possible_levels = getMaxLevels(width, height);
  if (level > max_possible_levels)
  {
    spdlog::warn("Requested level {} exceeds max possible levels {}", level, max_possible_levels);
    return;
  }
  
  size_t level_width = width >> level;
  size_t level_height = height >> level;
  
  if (level_width == 0 || level_height == 0)
  {
    return;
  }
  
  for (size_t y = 0; y < level_height; ++y)
  {
    const float* src = &data[y * width];
    float* dst = &output[y * level_width];
    std::copy(src, src + level_width, dst);
  }
}

void HaarWavelet2D::forwardSIMD(float* data, size_t width, size_t height, size_t levels)
{
  using namespace simd;
  
  if (levels == 0 || width < 2 || height < 2)
  {
    return;
  }
  
  size_t max_possible_levels = getMaxLevels(width, height);
  size_t actual_levels = std::min(levels, max_possible_levels);
  
  size_t current_width = width;
  size_t current_height = height;
  
  for (size_t level = 0; level < actual_levels; ++level)
  {
    if (current_width < 2 || current_height < 2)
    {
      break;
    }
    
    if (current_width >= SIMD_WIDTH * 2)
    {
      size_t batch_size = std::min(size_t(16), current_height);
      for (size_t y = 0; y < current_height; y += batch_size)
      {
        size_t rows_to_process = std::min(batch_size, current_height - y);
        
        for (size_t row = 0; row < rows_to_process; ++row)
        {
          batchHaarTransform1D(&data[(y + row) * width], 1, current_width);
        }
      }
    }
    else
    {
      for (size_t y = 0; y < current_height; ++y)
      {
        forward1D(&data[y * width], current_width, 1);
      }
    }
    
    if (current_height >= SIMD_WIDTH * 2)
    {
      size_t col_batch = std::min(size_t(32), current_width);
      
      auto col_buffer = allocateAligned<float>(current_height * col_batch);
      
      for (size_t x = 0; x < current_width; x += col_batch)
      {
        size_t cols_to_process = std::min(col_batch, current_width - x);
        
        for (size_t y = 0; y < current_height; ++y)
        {
          for (size_t col = 0; col < cols_to_process; ++col)
          {
            col_buffer[col * current_height + y] = data[y * width + x + col];
          }
        }
        
        batchHaarTransform1D(col_buffer.get(), cols_to_process, current_height);
        
        for (size_t y = 0; y < current_height; ++y)
        {
          for (size_t col = 0; col < cols_to_process; ++col)
          {
            data[y * width + x + col] = col_buffer[col * current_height + y];
          }
        }
      }
    }
    else
    {
      for (size_t x = 0; x < current_width; ++x)
      {
        forward1D(&data[x], current_height, width);
      }
    }
    
    current_width /= 2;
    current_height /= 2;
  }
}

void HaarWavelet2D::inverseSIMD(float* data, size_t width, size_t height, size_t levels)
{
  using namespace simd;
  
  if (levels == 0 || width < 2 || height < 2)
  {
    return;
  }
  
  size_t max_possible_levels = getMaxLevels(width, height);
  size_t actual_levels = std::min(levels, max_possible_levels);
  
  size_t current_width = width >> actual_levels;
  size_t current_height = height >> actual_levels;
  current_width = std::max(size_t(1), current_width);
  current_height = std::max(size_t(1), current_height);
  
  for (size_t level = actual_levels; level > 0; --level)
  {
    current_width = std::min(width, current_width * 2);
    current_height = std::min(height, current_height * 2);
    
    if (current_height >= SIMD_WIDTH * 2)
    {
      size_t col_batch = std::min(size_t(32), current_width);
      
      auto col_buffer = allocateAligned<float>(current_height * col_batch);
      
      for (size_t x = 0; x < current_width; x += col_batch)
      {
        size_t cols_to_process = std::min(col_batch, current_width - x);
        
        for (size_t y = 0; y < current_height; ++y)
        {
          for (size_t col = 0; col < cols_to_process; ++col)
          {
            col_buffer[col * current_height + y] = data[y * width + x + col];
          }
        }
        
        batchHaarInverse1D(col_buffer.get(), cols_to_process, current_height);
        
        for (size_t y = 0; y < current_height; ++y)
        {
          for (size_t col = 0; col < cols_to_process; ++col)
          {
            data[y * width + x + col] = col_buffer[col * current_height + y];
          }
        }
      }
    }
    else
    {
      for (size_t x = 0; x < current_width; ++x)
      {
        inverse1D(&data[x], current_height, width);
      }
    }
    
    if (current_width >= SIMD_WIDTH * 2)
    {
      size_t batch_size = std::min(size_t(16), current_height);
      for (size_t y = 0; y < current_height; y += batch_size)
      {
        size_t rows_to_process = std::min(batch_size, current_height - y);
        
        for (size_t row = 0; row < rows_to_process; ++row)
        {
          batchHaarInverse1D(&data[(y + row) * width], 1, current_width);
        }
      }
    }
    else
    {
      for (size_t y = 0; y < current_height; ++y)
      {
        inverse1D(&data[y * width], current_width, 1);
      }
    }
  }
}

void HaarWavelet2D::forward1D(float* data, size_t length, size_t stride)
{
  if (length < 2)
  {
    return;
  }
  
  size_t half_length = length / 2;
  
  if (stride == 1)
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    for (size_t i = 0; i < half_length; ++i)
    {
      const double a = static_cast<double>(data[2 * i]);
      const double b = static_cast<double>(data[2 * i + 1]);
      
      const double sum = a + b;
      const double diff = a - b;
      
      temp_buffer_[i] = static_cast<float>(sum * INV_SQRT2_DOUBLE);
      temp_buffer_[half_length + i] = static_cast<float>(diff * INV_SQRT2_DOUBLE);
    }
    
    std::copy(temp_buffer_.begin(), temp_buffer_.begin() + length, data);
  }
  else
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    for (size_t i = 0; i < length; ++i)
    {
      temp_buffer_[i] = data[i * stride];
    }
    
    for (size_t i = 0; i < half_length; ++i)
    {
      const double a = static_cast<double>(temp_buffer_[2 * i]);
      const double b = static_cast<double>(temp_buffer_[2 * i + 1]);
      
      const double sum = a + b;
      const double diff = a - b;
      
      data[i * stride] = static_cast<float>(sum * INV_SQRT2_DOUBLE);
      data[(half_length + i) * stride] = static_cast<float>(diff * INV_SQRT2_DOUBLE);
    }
  }
}

void HaarWavelet2D::inverse1D(float* data, size_t length, size_t stride)
{
  if (length < 2)
  {
    return;
  }
  
  size_t half_length = length / 2;
  
  if (stride == 1)
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    for (size_t i = 0; i < half_length; ++i)
    {
      const double avg = static_cast<double>(data[i]);
      const double diff = static_cast<double>(data[half_length + i]);
      
      const double scaled_avg = avg * INV_SQRT2_DOUBLE;
      const double scaled_diff = diff * INV_SQRT2_DOUBLE;
      
      temp_buffer_[2 * i] = static_cast<float>(scaled_avg + scaled_diff);
      temp_buffer_[2 * i + 1] = static_cast<float>(scaled_avg - scaled_diff);
    }
    
    std::copy(temp_buffer_.begin(), temp_buffer_.begin() + length, data);
  }
  else
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    for (size_t i = 0; i < length; ++i)
    {
      temp_buffer_[i] = data[i * stride];
    }
    
    for (size_t i = 0; i < half_length; ++i)
    {
      const double avg = static_cast<double>(temp_buffer_[i]);
      const double diff = static_cast<double>(temp_buffer_[half_length + i]);
      
      const double scaled_avg = avg * INV_SQRT2_DOUBLE;
      const double scaled_diff = diff * INV_SQRT2_DOUBLE;
      
      data[(2 * i) * stride] = static_cast<float>(scaled_avg + scaled_diff);
      data[(2 * i + 1) * stride] = static_cast<float>(scaled_avg - scaled_diff);
    }
  }
}

void HaarWavelet2D::forward1DSIMD(float* data, size_t length, size_t stride)
{
  if (stride == 1)
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    std::copy(data, data + length, temp_buffer_.begin());
    
    simd::batchHaarTransform1D(temp_buffer_.data(), 1, length);
    
    std::copy(temp_buffer_.begin(), temp_buffer_.begin() + length, data);
  }
  else
  {
    forward1D(data, length, stride);
  }
}

void HaarWavelet2D::inverse1DSIMD(float* data, size_t length, size_t stride)
{
  if (stride == 1)
  {
    if (temp_buffer_.size() < length)
    {
      temp_buffer_.resize(length);
    }
    
    std::copy(data, data + length, temp_buffer_.begin());
    
    simd::batchHaarInverse1D(temp_buffer_.data(), 1, length);
    
    std::copy(temp_buffer_.begin(), temp_buffer_.begin() + length, data);
  }
  else
  {
    inverse1D(data, length, stride);
  }
}

}
}
}