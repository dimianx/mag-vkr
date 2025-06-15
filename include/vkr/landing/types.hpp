#ifndef VKR_LANDING_TYPES_HPP_
#define VKR_LANDING_TYPES_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace vkr
{
namespace landing
{

struct RGBDFrame
{
  cv::Mat color;
  cv::Mat depth;
  double timestamp;
  
  cv::Mat K;
  cv::Mat D;
  
  Eigen::Isometry3f T_world_camera;
};

struct PointCloud
{
  std::vector<Eigen::Vector3f> points;
  std::vector<Eigen::Vector3f> normals;
  
  Eigen::Vector3f min_bound;
  Eigen::Vector3f max_bound;
};

struct LandingCandidate
{
  Eigen::Vector3f center;
  Eigen::Vector3f normal;
  Eigen::Vector4f plane_equation;
  
  std::vector<Eigen::Vector3f> boundary_points;
  std::vector<size_t> inlier_indices;
  
  float score = 0.0f;
  float area = 0.0f;
  float roughness = 0.0f;
  float tilt_angle = 0.0f;
  float confidence = 0.0f;
};

struct LandingResult
{
  std::vector<LandingCandidate> candidates;
  
  LandingCandidate best_candidate;
  bool has_valid_landing_zone = false;
  
  double computation_time_ms = 0.0;
};

struct LandingAnalysis
{
  bool zones_detected = false;
  size_t num_candidates = 0;
  bool has_safe_zone = false;
  
  float best_zone_score = 0.0f;
  
  bool ready_for_descent = false;
};

}
}

#endif