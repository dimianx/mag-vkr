#ifndef VKR_MATH_WAVELETS_WAVELET_TRANSFORM_2D_HPP_
#define VKR_MATH_WAVELETS_WAVELET_TRANSFORM_2D_HPP_

#include "vkr/math/wavelets/haar_wavelet_2d.hpp"
#include <memory>
#include <cstddef>

namespace vkr
{
namespace math
{
namespace wavelets
{

template<typename WaveletType>
class WaveletTransform2D
{
public:
  WaveletTransform2D();
  ~WaveletTransform2D() = default;
  
  void setWavelet(const WaveletType& wavelet);
  
  void forward(float* data, size_t width, size_t height, size_t levels);
  void inverse(float* data, size_t width, size_t height, size_t levels);
  
  struct Subband
  {
    float* data;
    size_t width;
    size_t height;
    size_t level;
    enum Type { LL, LH, HL, HH } type;
  };
  
  Subband getSubband(float* data, size_t width, size_t height,
                    size_t level, typename Subband::Type type) const;
  
  struct MinMax
  {
    float min;
    float max;
  };
  
  MinMax computeMinMax(const float* data, size_t width, size_t height,
                      size_t x, size_t y, size_t block_width, size_t block_height,
                      size_t levels) const;
  
  void batchForward(float** data_arrays, size_t count,
                   size_t width, size_t height, size_t levels);
  void batchInverse(float** data_arrays, size_t count,
                   size_t width, size_t height, size_t levels);
  
private:
  std::unique_ptr<WaveletType> wavelet_;
  
  size_t getSubbandOffset(size_t width, size_t height, size_t level,
                         typename Subband::Type type) const;
};

extern template class WaveletTransform2D<HaarWavelet2D>;

}
}
}

#endif