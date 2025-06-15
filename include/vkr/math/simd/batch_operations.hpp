#ifndef VKR_MATH_SIMD_BATCH_OPERATIONS_HPP_
#define VKR_MATH_SIMD_BATCH_OPERATIONS_HPP_

#include <Eigen/Core>
#include <cstddef>

namespace vkr
{
namespace math
{
namespace simd
{

void batchInterpolate(const float* heights, size_t count,
                     const float* u, const float* v,
                     float* output);

void batchDistanceCompute(const Eigen::Vector3f* points, size_t count,
                         const Eigen::Vector3f& query,
                         float* distances);

void batchSquaredDistanceCompute(const Eigen::Vector3f* points, size_t count,
                                const Eigen::Vector3f& query,
                                float* squared_distances);

void batchMin(const float* a, const float* b, size_t count, float* results);
void batchMax(const float* a, const float* b, size_t count, float* results);

void batchClamp(const float* values, size_t count,
                float min_value, float max_value,
                float* results);

void batchPolynomialEval(const float* coefficients, int degree,
                        const float* t_values, size_t count,
                        float* results);

void batchHaarTransform1D(float* data, size_t batch_size, size_t length);
void batchHaarInverse1D(float* data, size_t batch_size, size_t length);

}
}
}

#endif