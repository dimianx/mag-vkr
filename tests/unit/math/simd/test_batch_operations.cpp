#include <gtest/gtest.h>
#include "vkr/math/simd/batch_operations.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <Eigen/Core>
#include <random>
#include <vector>
#include <cmath>
#include <limits>

namespace vkr 
{
namespace math 
{
namespace simd 
{

class BatchOperationsTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    rng_.seed(42);
    
    static bool logged = false;
    if (!logged) 
    {
      logSIMDInfo();
      logged = true;
    }
  }
  
  std::mt19937 rng_;
  std::uniform_real_distribution<float> value_dist_{-10.0f, 10.0f};
  std::uniform_real_distribution<float> unit_dist_{0.0f, 1.0f};
  std::uniform_real_distribution<float> extended_dist_{-2.0f, 2.0f};
};

TEST_F(BatchOperationsTest, BatchInterpolate) 
{
  constexpr size_t count = 257;
  
  std::vector<float> heights(count * 4);
  std::vector<float> u(count), v(count);
  std::vector<float> output(count);
  std::vector<float> output_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    for (size_t j = 0; j < 4; ++j) 
    {
      heights[i * 4 + j] = value_dist_(rng_);
    }
    u[i] = unit_dist_(rng_);
    v[i] = unit_dist_(rng_);
    
    float h00 = heights[i * 4 + 0];
    float h10 = heights[i * 4 + 1];
    float h01 = heights[i * 4 + 2];
    float h11 = heights[i * 4 + 3];
    
    output_ref[i] = (1.0f - u[i]) * (1.0f - v[i]) * h00 +
                    u[i] * (1.0f - v[i]) * h10 +
                    (1.0f - u[i]) * v[i] * h01 +
                    u[i] * v[i] * h11;
  }
  
  batchInterpolate(heights.data(), count, u.data(), v.data(), output.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(output[i], output_ref[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchInterpolateExtrapolation) 
{
  constexpr size_t count = 128;
  
  std::vector<float> heights(count * 4);
  std::vector<float> u(count), v(count);
  std::vector<float> output(count);
  std::vector<float> output_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    for (size_t j = 0; j < 4; ++j) 
    {
      heights[i * 4 + j] = value_dist_(rng_);
    }
    u[i] = extended_dist_(rng_);
    v[i] = extended_dist_(rng_);
    
    float h00 = heights[i * 4 + 0];
    float h10 = heights[i * 4 + 1];
    float h01 = heights[i * 4 + 2];
    float h11 = heights[i * 4 + 3];
    
    output_ref[i] = (1.0f - u[i]) * (1.0f - v[i]) * h00 +
                    u[i] * (1.0f - v[i]) * h10 +
                    (1.0f - u[i]) * v[i] * h01 +
                    u[i] * v[i] * h11;
  }
  
  batchInterpolate(heights.data(), count, u.data(), v.data(), output.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(output[i], output_ref[i], 1e-4f) 
      << "Extrapolation mismatch at index " << i 
      << " with u=" << u[i] << ", v=" << v[i];
  }
}

TEST_F(BatchOperationsTest, BatchDistanceCompute) 
{
  constexpr size_t count = 1024;
  std::vector<Eigen::Vector3f> points(count);
  Eigen::Vector3f query(value_dist_(rng_), value_dist_(rng_), value_dist_(rng_));
  std::vector<float> distances(count);
  std::vector<float> distances_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    points[i] = Eigen::Vector3f(value_dist_(rng_), value_dist_(rng_), value_dist_(rng_));
    distances_ref[i] = (points[i] - query).norm();
  }
  
  batchDistanceCompute(points.data(), count, query, distances.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(distances[i], distances_ref[i], 1e-5f) << "Mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchSquaredDistanceCompute) 
{
  constexpr size_t count = 513;
  std::vector<Eigen::Vector3f> points(count);
  Eigen::Vector3f query(value_dist_(rng_), value_dist_(rng_), value_dist_(rng_));
  std::vector<float> squared_distances(count);
  std::vector<float> squared_distances_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    points[i] = Eigen::Vector3f(value_dist_(rng_), value_dist_(rng_), value_dist_(rng_));
    squared_distances_ref[i] = (points[i] - query).squaredNorm();
  }
  
  batchSquaredDistanceCompute(points.data(), count, query, squared_distances.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(squared_distances[i], squared_distances_ref[i], 1e-5f) 
      << "Mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchMinMax) 
{
  constexpr size_t count = 2048;
  std::vector<float> a(count), b(count);
  std::vector<float> min_results(count), max_results(count);
  std::vector<float> min_ref(count), max_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    a[i] = value_dist_(rng_);
    b[i] = value_dist_(rng_);
    min_ref[i] = std::min(a[i], b[i]);
    max_ref[i] = std::max(a[i], b[i]);
  }
  
  batchMin(a.data(), b.data(), count, min_results.data());
  batchMax(a.data(), b.data(), count, max_results.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(min_results[i], min_ref[i]) << "Min mismatch at index " << i;
    EXPECT_EQ(max_results[i], max_ref[i]) << "Max mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchClamp) 
{
  constexpr size_t count = 1337;
  std::vector<float> values(count);
  std::vector<float> results(count);
  std::vector<float> results_ref(count);
  
  float min_val = -5.0f;
  float max_val = 5.0f;
  
  for (size_t i = 0; i < count; ++i) 
  {
    values[i] = value_dist_(rng_) * 2.0f;
    results_ref[i] = std::max(min_val, std::min(max_val, values[i]));
  }
  
  batchClamp(values.data(), count, min_val, max_val, results.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(results[i], results_ref[i]) << "Mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchPolynomialEval) 
{
  constexpr size_t count = 768;
  constexpr int degree = 5;
  
  float coefficients[degree + 1] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  
  std::vector<float> t_values(count);
  std::vector<float> results(count);
  std::vector<float> results_ref(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    t_values[i] = unit_dist_(rng_) * 2.0f - 1.0f;
    
    float t = t_values[i];
    float result = coefficients[degree];
    for (int j = degree - 1; j >= 0; --j) 
    {
      result = result * t + coefficients[j];
    }
    results_ref[i] = result;
  }
  
  batchPolynomialEval(coefficients, degree, t_values.data(), count, results.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(results[i], results_ref[i], 1e-4f) << "Mismatch at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchPolynomialHighDegree) 
{
  constexpr size_t count = 256;
  constexpr int degree = 15;
  
  float coefficients[degree + 1];
  for (int i = 0; i <= degree; ++i) 
  {
    coefficients[i] = 1.0f / (i + 1);
  }
  
  std::vector<float> t_values(count);
  std::vector<float> results(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    t_values[i] = unit_dist_(rng_) * 0.8f - 0.4f;
  }
  
  batchPolynomialEval(coefficients, degree, t_values.data(), count, results.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_TRUE(std::isfinite(results[i])) 
      << "Non-finite result at index " << i << " for t=" << t_values[i];
    
    EXPECT_LT(std::abs(results[i]), 1e6f) 
      << "Result too large at index " << i;
  }
}

TEST_F(BatchOperationsTest, BatchHaarTransform) 
{
  constexpr size_t batch_size = 16;
  constexpr size_t length = 64;
  
  std::vector<float> data(batch_size * length);
  std::vector<float> data_backup(batch_size * length);
  
  for (size_t b = 0; b < batch_size; ++b) 
  {
    for (size_t i = 0; i < length; ++i) 
    {
      float x = static_cast<float>(i) / length;
      data[b * length + i] = std::sin(2.0f * M_PI * x) + 
                             0.5f * std::cos(4.0f * M_PI * x);
    }
  }
  
  data_backup = data;
  
  batchHaarTransform1D(data.data(), batch_size, length);
  
  bool changed = false;
  for (size_t i = 0; i < data.size(); ++i) 
  {
    if (std::abs(data[i] - data_backup[i]) > 1e-6f) 
    {
      changed = true;
      break;
    }
  }
  EXPECT_TRUE(changed) << "Transform should modify data";
  
  batchHaarInverse1D(data.data(), batch_size, length);
  
  for (size_t i = 0; i < data.size(); ++i) 
  {
    EXPECT_NEAR(data[i], data_backup[i], 1e-5f) 
      << "Reconstruction error at index " << i;
  }
}

TEST_F(BatchOperationsTest, EdgeCases) 
{
  {
    std::vector<float> empty;
    std::vector<float> result;
    ASSERT_NO_THROW(batchMin(empty.data(), empty.data(), 0, result.data()));
  }
  
  {
    float a = 5.0f, b = 3.0f, result;
    batchMin(&a, &b, 1, &result);
    EXPECT_EQ(result, 3.0f);
  }
}

TEST_F(BatchOperationsTest, UnalignedDataOperations) 
{
  constexpr size_t count = 100;
  constexpr size_t offset = 1;
  
  std::vector<uint8_t> buffer_a(sizeof(float) * (count + 16) + offset);
  std::vector<uint8_t> buffer_b(sizeof(float) * (count + 16) + offset);
  std::vector<uint8_t> buffer_result(sizeof(float) * (count + 16) + offset);
  
  float* a = reinterpret_cast<float*>(buffer_a.data() + offset);
  float* b = reinterpret_cast<float*>(buffer_b.data() + offset);
  float* result = reinterpret_cast<float*>(buffer_result.data() + offset);
  
  for (size_t i = 0; i < count; ++i) 
  {
    a[i] = value_dist_(rng_);
    b[i] = value_dist_(rng_);
  }
  
  ASSERT_NO_THROW(batchMin(a, b, count, result));
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_EQ(result[i], std::min(a[i], b[i])) 
      << "Unaligned min failed at index " << i;
  }
}

TEST_F(BatchOperationsTest, ArrayEndMasking) 
{
  std::vector<size_t> test_sizes = {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33};
  
  for (size_t count : test_sizes) 
  {
    std::vector<float> a(count), b(count);
    std::vector<float> min_result(count);
    std::vector<float> max_result(count);
    
    for (size_t i = 0; i < count; ++i) 
    {
      a[i] = static_cast<float>(i);
      b[i] = static_cast<float>(count - i);
    }
    
    batchMin(a.data(), b.data(), count, min_result.data());
    batchMax(a.data(), b.data(), count, max_result.data());
    
    for (size_t i = 0; i < count; ++i) 
    {
      EXPECT_EQ(min_result[i], std::min(a[i], b[i])) 
        << "Min failed for count=" << count << " at index " << i;
      EXPECT_EQ(max_result[i], std::max(a[i], b[i])) 
        << "Max failed for count=" << count << " at index " << i;
    }
  }
}

TEST_F(BatchOperationsTest, SpecialFloatValues) 
{
  constexpr size_t count = 32;
  std::vector<float> a(count), b(count);
  std::vector<float> result(count);
  
  a[0] = std::numeric_limits<float>::quiet_NaN();
  b[0] = 1.0f;
  
  a[1] = std::numeric_limits<float>::infinity();
  b[1] = 2.0f;
  
  a[2] = -std::numeric_limits<float>::infinity();
  b[2] = 3.0f;
  
  a[3] = std::numeric_limits<float>::min();
  b[3] = std::numeric_limits<float>::max();
  
  for (size_t i = 4; i < count; ++i) 
  {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(count - i);
  }
  
  batchMin(a.data(), b.data(), count, result.data());
  
  EXPECT_TRUE(std::isnan(result[0])) << "NaN handling failed in min";
  EXPECT_EQ(result[1], 2.0f) << "Inf handling failed in min";
  EXPECT_EQ(result[2], -std::numeric_limits<float>::infinity()) << "-Inf handling failed";
  
  constexpr int degree = 3;
  float coeffs[degree + 1] = {1.0f, 0.5f, 0.25f, 0.125f};
  std::vector<float> t_values(count);
  
  t_values[0] = std::numeric_limits<float>::quiet_NaN();
  t_values[1] = std::numeric_limits<float>::infinity();
  t_values[2] = -std::numeric_limits<float>::infinity();
  t_values[3] = 0.0f;
  
  for (size_t i = 4; i < count; ++i) 
  {
    t_values[i] = unit_dist_(rng_);
  }
  
  batchPolynomialEval(coeffs, degree, t_values.data(), count, result.data());
  
  EXPECT_TRUE(std::isnan(result[0])) << "NaN should propagate through polynomial";
  EXPECT_TRUE(std::isinf(result[1])) << "Inf should propagate through polynomial";
  EXPECT_TRUE(std::isinf(result[2])) << "-Inf should propagate through polynomial";
  EXPECT_EQ(result[3], coeffs[0]) << "t=0 should give first coefficient";
}

TEST_F(BatchOperationsTest, WaveletTransformCompleteness) 
{
  constexpr size_t batch_size = 8;
  constexpr size_t length = 128;
  
  std::vector<float> data(batch_size * length);
  std::vector<float> data_backup(batch_size * length);
  
  for (size_t b = 0; b < batch_size; ++b) 
  {
    for (size_t i = 0; i < length; ++i) 
    {
      float x = static_cast<float>(i) / length;
      
      switch (b % 4) 
      {
        case 0:
          data[b * length + i] = std::sin(2.0f * M_PI * x);
          break;
        case 1:
          data[b * length + i] = (i < length/2) ? 1.0f : -1.0f;
          break;
        case 2:
          data[b * length + i] = x;
          break;
        case 3:
          data[b * length + i] = value_dist_(rng_);
          break;
      }
    }
  }
  
  data_backup = data;
  
  batchHaarTransform1D(data.data(), batch_size, length);
  
  for (size_t b = 0; b < batch_size; ++b) 
  {
    float energy_original = 0.0f;
    float energy_transformed = 0.0f;
    
    for (size_t i = 0; i < length; ++i) 
    {
      energy_original += data_backup[b * length + i] * data_backup[b * length + i];
      energy_transformed += data[b * length + i] * data[b * length + i];
    }
    
    EXPECT_NEAR(energy_transformed, energy_original, energy_original * 1e-4f) 
      << "Energy not preserved for batch " << b;
  }
  
  batchHaarInverse1D(data.data(), batch_size, length);
  
  for (size_t i = 0; i < data.size(); ++i) 
  {
    EXPECT_NEAR(data[i], data_backup[i], 1e-5f) 
      << "Reconstruction error at index " << i;
  }
}

TEST_F(BatchOperationsTest, ExtremeMagnitudes) 
{
  constexpr size_t count = 64;
  std::vector<Eigen::Vector3f> points(count);
  std::vector<float> distances(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    float scale = std::pow(10.0f, static_cast<float>(i - 32));
    points[i] = Eigen::Vector3f(scale, 0, 0);
  }
  
  Eigen::Vector3f query(0, 0, 0);
  
  ASSERT_NO_THROW(batchDistanceCompute(points.data(), count, query, distances.data()));
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_TRUE(std::isfinite(distances[i]) || std::isinf(distances[i])) 
      << "Invalid distance at index " << i;
    
    if (std::isfinite(distances[i])) 
    {
      float expected = points[i].norm();
      EXPECT_NEAR(distances[i], expected, expected * 1e-5f) 
        << "Distance mismatch at index " << i;
    }
  }
}

TEST_F(BatchOperationsTest, StridedDataOperations) 
{
  constexpr size_t count = 100;
  constexpr size_t stride = 3;
  
  std::vector<float> data(count * stride);
  std::vector<float> result(count);
  
  for (size_t i = 0; i < count; ++i) 
  {
    data[i * stride] = static_cast<float>(i);
  }
  
  std::vector<float> contiguous(count);
  for (size_t i = 0; i < count; ++i) 
  {
    contiguous[i] = data[i * stride];
  }
  
  float coeffs[3] = {1.0f, 2.0f, 3.0f};
  batchPolynomialEval(coeffs, 2, contiguous.data(), count, result.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    float t = contiguous[i];
    float expected = coeffs[0] + coeffs[1] * t + coeffs[2] * t * t;
    EXPECT_NEAR(result[i], expected, 1e-5f) 
      << "Polynomial eval mismatch at index " << i;
  }
}

}
}
}