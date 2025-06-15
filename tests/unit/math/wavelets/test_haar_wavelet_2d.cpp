#include <gtest/gtest.h>
#include "vkr/math/wavelets/haar_wavelet_2d.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

namespace vkr 
{
namespace math 
{
namespace wavelets 
{

class HaarWavelet2DTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    rng_.seed(42);
    wavelet_ = std::make_unique<HaarWavelet2D>();
  }
  
  std::vector<float> createTestData(size_t width, size_t height, 
                                   std::function<float(size_t, size_t)> generator) 
  {
    std::vector<float> data(width * height);
    for (size_t y = 0; y < height; ++y) 
    {
      for (size_t x = 0; x < width; ++x) 
      {
        data[y * width + x] = generator(x, y);
      }
    }
    return data;
  }
  
  float computeMaxDifference(const std::vector<float>& a, const std::vector<float>& b) 
  {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) 
    {
      max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
  }
  
  std::mt19937 rng_;
  std::uniform_real_distribution<float> value_dist_{-100.0f, 100.0f};
  std::unique_ptr<HaarWavelet2D> wavelet_;
};

TEST_F(HaarWavelet2DTest, ForwardInverseTransform) 
{
  std::vector<size_t> sizes = {4, 8, 16, 32, 64, 128};
  
  for (size_t size : sizes) 
  {
    auto original = createTestData(size, size, [this](size_t, size_t) 
    {
      return value_dist_(rng_);
    });
    
    auto data = original;
    
    wavelet_->forward(data.data(), size, size, 3);
    
    EXPECT_NE(data, original);
    
    wavelet_->inverse(data.data(), size, size, 3);
    
    float max_diff = computeMaxDifference(data, original);
    EXPECT_LT(max_diff, 1e-5f) << "Size: " << size;
  }
}

TEST_F(HaarWavelet2DTest, ConstantData) 
{
  constexpr size_t size = 32;
  constexpr float value = 42.0f;
  
  auto data = createTestData(size, size, [value](size_t, size_t) 
  {
    return value;
  });
  
  wavelet_->forward(data.data(), size, size, 5);
  
  EXPECT_NEAR(data[0], value * size, 1e-5f);
  
  for (size_t i = 1; i < data.size(); ++i) 
  {
    EXPECT_NEAR(data[i], 0.0f, 1e-5f);
  }
}

TEST_F(HaarWavelet2DTest, LinearGradient) 
{
  constexpr size_t size = 64;
  
  auto h_gradient = createTestData(size, size, [](size_t x, size_t) 
  {
    return static_cast<float>(x);
  });
  
  auto data = h_gradient;
  wavelet_->forward(data.data(), size, size, 4);
  wavelet_->inverse(data.data(), size, size, 4);
  
  float max_diff = computeMaxDifference(data, h_gradient);
  EXPECT_LT(max_diff, 1e-4f);
  
  auto v_gradient = createTestData(size, size, [](size_t, size_t y) 
  {
    return static_cast<float>(y);
  });
  
  data = v_gradient;
  wavelet_->forward(data.data(), size, size, 4);
  wavelet_->inverse(data.data(), size, size, 4);
  
  max_diff = computeMaxDifference(data, v_gradient);
  EXPECT_LT(max_diff, 1e-4f);
}

TEST_F(HaarWavelet2DTest, Quantization) 
{
  constexpr size_t size = 32;
  constexpr float quant_step = 0.1f;
  
  auto original = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  auto data = original;
  
  wavelet_->forwardQuantized(data.data(), size, size, 3, quant_step);
  
  for (size_t i = 0; i < data.size(); ++i) 
  {
    if (std::abs(data[i]) > 1e-9f) 
    {
      const double scaled_value = static_cast<double>(data[i]) / quant_step;
      const double rounded_value = std::round(scaled_value);
      EXPECT_NEAR(scaled_value, rounded_value, 1e-4)
        << "Value " << data[i] << " is not a multiple of " << quant_step;
    }
  }
  
  wavelet_->inverse(data.data(), size, size, 3);
  
  float max_diff = computeMaxDifference(data, original);
  EXPECT_LT(max_diff, size * quant_step);
}

TEST_F(HaarWavelet2DTest, QuantizationWithError) 
{
  constexpr size_t size = 16;
  std::vector<float> quant_steps = {0.01f, 0.1f, 1.0f, 10.0f};
  
  auto original = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  for (float quant_step : quant_steps) 
  {
    auto data = original;
    
    float reported_error = wavelet_->forwardQuantizedWithError(
      data.data(), size, size, 3, quant_step);
    
    wavelet_->inverse(data.data(), size, size, 3);
    float actual_error = computeMaxDifference(data, original);
    
    EXPECT_NEAR(reported_error, actual_error, 1e-5f) 
      << "Quantization step: " << quant_step;
    
    EXPECT_LT(actual_error, size * quant_step);
  }
}

TEST_F(HaarWavelet2DTest, DifferentLevels) 
{
  constexpr size_t size = 64;
  
  auto original = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  for (size_t levels = 0; levels <= 6; ++levels) 
  {
    auto data = original;
    
    if (levels == 0) 
    {
      wavelet_->forward(data.data(), size, size, levels);
      EXPECT_EQ(data, original);
    }
    else 
    {
      wavelet_->forward(data.data(), size, size, levels);
      wavelet_->inverse(data.data(), size, size, levels);
      
      float max_diff = computeMaxDifference(data, original);
      EXPECT_LT(max_diff, 1e-4f) << "Levels: " << levels;
    }
  }
}

TEST_F(HaarWavelet2DTest, GetCoefficientsAtLevel) 
{
  constexpr size_t size = 32;
  
  auto data = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  wavelet_->forward(data.data(), size, size, 3);
  
  for (size_t level = 0; level < 3; ++level) 
  {
    size_t level_size = size >> level;
    std::vector<float> coeffs(level_size * level_size);
    
    wavelet_->getCoefficientsAtLevel(data.data(), size, size, level, coeffs.data());
    
    EXPECT_EQ(coeffs.size(), level_size * level_size);
    
    bool all_zero = true;
    for (float c : coeffs) 
    {
      if (std::abs(c) > 1e-6f) 
      {
        all_zero = false;
        break;
      }
    }
    EXPECT_FALSE(all_zero) << "Level: " << level;
  }
}

TEST_F(HaarWavelet2DTest, SIMDVersions) 
{
  constexpr size_t size = 128;
  
  auto original = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  auto data_regular = original;
  wavelet_->forward(data_regular.data(), size, size, 4);
  wavelet_->inverse(data_regular.data(), size, size, 4);
  
  auto data_simd = original;
  wavelet_->forwardSIMD(data_simd.data(), size, size, 4);
  wavelet_->inverseSIMD(data_simd.data(), size, size, 4);
  
  float max_diff = computeMaxDifference(data_regular, data_simd);
  EXPECT_LT(max_diff, 1e-4f);
}

TEST_F(HaarWavelet2DTest, EdgeCases) 
{
  {
    std::vector<float> tiny_data = {1.0f, 2.0f, 3.0f, 4.0f};
    wavelet_->forward(tiny_data.data(), 2, 2, 1);
    
    float sum = tiny_data[0];
    EXPECT_NEAR(sum, 10.0f / 2.0f, 1e-5f);
  }
  
  {
    constexpr size_t width = 32;
    constexpr size_t height = 64;
    
    auto data = createTestData(width, height, [this](size_t, size_t) 
    {
      return value_dist_(rng_);
    });
    
    auto original = data;
    
    wavelet_->forward(data.data(), width, height, 3);
    wavelet_->inverse(data.data(), width, height, 3);
    
    float max_diff = computeMaxDifference(data, original);
    EXPECT_LT(max_diff, 1e-4f);
  }
  
  {
    constexpr size_t size = 16;
    std::vector<float> zero_data(size * size, 0.0f);
    
    wavelet_->forward(zero_data.data(), size, size, 2);
    
    for (float val : zero_data) 
    {
      EXPECT_EQ(val, 0.0f);
    }
  }
}

TEST_F(HaarWavelet2DTest, BoundaryHandling) 
{
  constexpr size_t size = 16;
  auto data = createTestData(size, size, [](size_t x, size_t y) 
  {
    if (x == 0 || y == 0 || x == size - 1 || y == size - 1) 
    {
      return 100.0f;
    }
    return 0.0f;
  });
  
  auto original = data;
  
  wavelet_->forward(data.data(), size, size, 3);
  wavelet_->inverse(data.data(), size, size, 3);
  
  for (size_t y = 0; y < size; ++y) 
  {
    for (size_t x = 0; x < size; ++x) 
    {
      size_t idx = y * size + x;
      if (x == 0 || y == 0 || x == size - 1 || y == size - 1) 
      {
        EXPECT_NEAR(data[idx], 100.0f, 1e-4f);
      }
      else 
      {
        EXPECT_NEAR(data[idx], 0.0f, 1e-4f);
      }
    }
  }
}

TEST_F(HaarWavelet2DTest, LargeData) 
{
  constexpr size_t size = 512;
  
  auto data = createTestData(size, size, [this](size_t, size_t) 
  {
    return value_dist_(rng_);
  });
  
  auto original = data;
  
  ASSERT_NO_THROW(wavelet_->forward(data.data(), size, size, 6));
  ASSERT_NO_THROW(wavelet_->inverse(data.data(), size, size, 6));
  
  float max_diff = computeMaxDifference(data, original);
  EXPECT_LT(max_diff, 1e-3f);
}

}
}
}