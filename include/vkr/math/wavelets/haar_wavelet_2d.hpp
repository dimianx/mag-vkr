#ifndef VKR_MATH_WAVELETS_HAAR_WAVELET_2D_HPP_
#define VKR_MATH_WAVELETS_HAAR_WAVELET_2D_HPP_

#include <cstddef>
#include <vector>

namespace vkr
{
namespace math
{
namespace wavelets
{

class HaarWavelet2D
{
public:
  HaarWavelet2D() = default;
  ~HaarWavelet2D() = default;
  
  void forward(float* data, size_t width, size_t height, size_t levels);
  
  void inverse(float* data, size_t width, size_t height, size_t levels);
  
  void forwardQuantized(float* data, size_t width, size_t height,
                       size_t levels, float quantization_step);
  
  float forwardQuantizedWithError(float* data, size_t width, size_t height,
                                 size_t levels, float quantization_step);
  
  void getCoefficientsAtLevel(const float* data, size_t width, size_t height,
                             size_t level, float* output) const;
  
  void forwardSIMD(float* data, size_t width, size_t height, size_t levels);
  void inverseSIMD(float* data, size_t width, size_t height, size_t levels);
  
private:
  void forward1D(float* data, size_t length, size_t stride);
  void inverse1D(float* data, size_t length, size_t stride);
  
  void forward1DSIMD(float* data, size_t length, size_t stride);
  void inverse1DSIMD(float* data, size_t length, size_t stride);
  
  std::vector<float> temp_buffer_;
};

}
}
}

#endif