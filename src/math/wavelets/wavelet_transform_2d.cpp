#include "vkr/math/wavelets/wavelet_transform_2d.hpp"
#include <algorithm>
#include <limits>
#include <cstring>

namespace vkr
{
namespace math
{
namespace wavelets
{

template<typename WaveletType>
WaveletTransform2D<WaveletType>::WaveletTransform2D()
  : wavelet_(std::make_unique<WaveletType>())
{
}

template<typename WaveletType>
void WaveletTransform2D<WaveletType>::setWavelet(const WaveletType& wavelet)
{
  wavelet_ = std::make_unique<WaveletType>(wavelet);
}

template<typename WaveletType>
void WaveletTransform2D<WaveletType>::forward(float* data, size_t width, size_t height, size_t levels)
{
  if (!wavelet_) 
  {
    wavelet_ = std::make_unique<WaveletType>();
  }
  
  wavelet_->forward(data, width, height, levels);
}

template<typename WaveletType>
void WaveletTransform2D<WaveletType>::inverse(float* data, size_t width, size_t height, size_t levels)
{
  if (!wavelet_) 
  {
    wavelet_ = std::make_unique<WaveletType>();
  }
  
  wavelet_->inverse(data, width, height, levels);
}

template<typename WaveletType>
typename WaveletTransform2D<WaveletType>::Subband 
WaveletTransform2D<WaveletType>::getSubband(float* data, size_t width, size_t height,
                                            size_t level, typename Subband::Type type) const
{
  Subband result;
  
  size_t level_width = width >> level;
  size_t level_height = height >> level;
  
  result.level = level;
  result.type = type;
  
  switch (type) 
  {
    case Subband::Type::LL:
      result.data = data;
      result.width = level_width;
      result.height = level_height;
      break;
      
    case Subband::Type::LH:
      result.data = data + level_width;
      result.width = level_width;
      result.height = level_height;
      break;
      
    case Subband::Type::HL:
      result.data = data + level_height * width;
      result.width = level_width;
      result.height = level_height;
      break;
      
    case Subband::Type::HH:
      result.data = data + level_height * width + level_width;
      result.width = level_width;
      result.height = level_height;
      break;
  }
  
  return result;
}

template<typename WaveletType>
size_t WaveletTransform2D<WaveletType>::getSubbandOffset(size_t width, size_t height, size_t level,
                                                         typename Subband::Type type) const
{
  size_t level_width = width >> level;
  size_t level_height = height >> level;
  
  switch (type) 
  {
    case Subband::Type::LL:
      return 0;
    case Subband::Type::LH:
      return level_width;
    case Subband::Type::HL:
      return level_height * width;
    case Subband::Type::HH:
      return level_height * width + level_width;
    default:
      return 0;
  }
}

template<typename WaveletType>
typename WaveletTransform2D<WaveletType>::MinMax 
WaveletTransform2D<WaveletType>::computeMinMax(const float* data, size_t width, size_t height,
                                               size_t x, size_t y, size_t block_width, size_t block_height,
                                               size_t levels) const
{
  MinMax result;
  result.min = std::numeric_limits<float>::max();
  result.max = std::numeric_limits<float>::lowest();
  
  block_width = std::min(block_width, width - x);
  block_height = std::min(block_height, height - y);
  
  for (size_t level = 0; level <= levels; ++level) 
  {
    size_t level_factor = 1 << level;
    size_t level_x = x / level_factor;
    size_t level_y = y / level_factor;
    size_t level_block_width = (block_width + level_factor - 1) / level_factor;
    size_t level_block_height = (block_height + level_factor - 1) / level_factor;
    
    for (int subband = 0; subband < 4; ++subband) 
    {
      typename Subband::Type type = static_cast<typename Subband::Type>(subband);
      
      if (type == Subband::Type::LL && level < levels) continue;
      
      size_t offset = getSubbandOffset(width, height, level, type);
      
      for (size_t dy = 0; dy < level_block_height; ++dy) 
      {
        for (size_t dx = 0; dx < level_block_width; ++dx) 
        {
          size_t px = level_x + dx;
          size_t py = level_y + dy;
          
          if (px < (width >> level) && py < (height >> level)) 
          {
            size_t idx = offset + py * width + px;
            float value = data[idx];
            result.min = std::min(result.min, value);
            result.max = std::max(result.max, value);
          }
        }
      }
    }
  }
  
  return result;
}

template<typename WaveletType>
void WaveletTransform2D<WaveletType>::batchForward(float** data_arrays, size_t count,
                                                   size_t width, size_t height, size_t levels)
{
  if (!wavelet_) 
  {
    wavelet_ = std::make_unique<WaveletType>();
  }
  
  for (size_t i = 0; i < count; ++i) 
  {
    if (data_arrays[i]) 
    {
      if constexpr (std::is_same_v<WaveletType, HaarWavelet2D>) 
      {
        wavelet_->forwardSIMD(data_arrays[i], width, height, levels);
      }
      else 
      {
        wavelet_->forward(data_arrays[i], width, height, levels);
      }
    }
  }
}

template<typename WaveletType>
void WaveletTransform2D<WaveletType>::batchInverse(float** data_arrays, size_t count,
                                                   size_t width, size_t height, size_t levels)
{
  if (!wavelet_) 
  {
    wavelet_ = std::make_unique<WaveletType>();
  }
  
  for (size_t i = 0; i < count; ++i) 
  {
    if (data_arrays[i]) 
    {
      if constexpr (std::is_same_v<WaveletType, HaarWavelet2D>) 
      {
        wavelet_->inverseSIMD(data_arrays[i], width, height, levels);
      }
      else 
      {
        wavelet_->inverse(data_arrays[i], width, height, levels);
      }
    }
  }
}

template class WaveletTransform2D<HaarWavelet2D>;

}
}
}