#include "vkr/landing/landing_unit.hpp"
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <chrono>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <set>

namespace vkr 
{
namespace landing 
{

namespace 
{

constexpr float DEG_TO_RAD = M_PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / M_PI;

struct PlaneRANSAC
{
  Eigen::Vector4f coefficients;
  std::vector<size_t> inliers;
  float score;
};

Eigen::Vector4f computePlane(const Eigen::Vector3f& p1, 
                            const Eigen::Vector3f& p2, 
                            const Eigen::Vector3f& p3)
{
  Eigen::Vector3f v1 = p2 - p1;
  Eigen::Vector3f v2 = p3 - p1;
  Eigen::Vector3f normal = v1.cross(v2).normalized();
  float d = -normal.dot(p1);
  return Eigen::Vector4f(normal.x(), normal.y(), normal.z(), d);
}

float pointToPlaneDistance(const Eigen::Vector3f& point, 
                          const Eigen::Vector4f& plane)
{
  return std::abs(plane.head<3>().dot(point) + plane.w()) / 
         plane.head<3>().norm();
}

Eigen::Vector3f projectOntoPlane(const Eigen::Vector3f& point,
                                const Eigen::Vector4f& plane)
{
  Eigen::Vector3f normal = plane.head<3>().normalized();
  float dist = plane.head<3>().dot(point) + plane.w();
  return point - dist * normal / plane.head<3>().norm();
}

std::vector<size_t> convexHull2D(const std::vector<Eigen::Vector2f>& points)
{
  if (points.size() < 3) 
  {
    return {};
  }
  
  size_t start = 0;
  for (size_t i = 1; i < points.size(); ++i)
  {
    if (points[i].x() < points[start].x() ||
        (points[i].x() == points[start].x() && points[i].y() < points[start].y()))
    {
      start = i;
    }
  }
  
  std::vector<size_t> indices(points.size());
  std::iota(indices.begin(), indices.end(), 0);
  
  auto polarAngle = [&](size_t i) -> float
  {
    if (i == start) 
    {
      return -static_cast<float>(M_PI);
    }
    return std::atan2(points[i].y() - points[start].y(),
                     points[i].x() - points[start].x());
  };
  
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) 
  {
    return polarAngle(a) < polarAngle(b);
  });
  
  std::vector<size_t> hull;
  for (size_t idx : indices)
  {
    while (hull.size() > 1)
    {
      size_t last = hull.back();
      size_t prev = hull[hull.size() - 2];
      
      Eigen::Vector2f v1 = points[last] - points[prev];
      Eigen::Vector2f v2 = points[idx] - points[last];
      
      if (v1.x() * v2.y() - v1.y() * v2.x() <= 0)
      {
        hull.pop_back();
      }
      else
      {
        break;
      }
    }
    hull.push_back(idx);
  }
  
  return hull;
}

float computePolygonArea(const std::vector<Eigen::Vector2f>& points)
{
  if (points.size() < 3) 
  {
    return 0.0f;
  }
  
  float area = 0.0f;
  for (size_t i = 0; i < points.size(); ++i)
  {
    size_t j = (i + 1) % points.size();
    area += points[i].x() * points[j].y();
    area -= points[j].x() * points[i].y();
  }
  
  return std::abs(area) * 0.5f;
}

struct GridCell
{
  std::vector<size_t> point_indices;
  bool is_occupied = false;
};

float computeAreaWithHoles(const std::vector<Eigen::Vector2f>& points, float grid_resolution = 0.05f)
{
  if (points.size() < 3)
  {
    return 0.0f;
  }
  
  Eigen::Vector2f min_pt = points[0];
  Eigen::Vector2f max_pt = points[0];
  
  for (const auto& pt : points)
  {
    min_pt = min_pt.cwiseMin(pt);
    max_pt = max_pt.cwiseMax(pt);
  }
  
  min_pt -= Eigen::Vector2f::Constant(grid_resolution);
  max_pt += Eigen::Vector2f::Constant(grid_resolution);
  
  int grid_width = static_cast<int>((max_pt.x() - min_pt.x()) / grid_resolution) + 1;
  int grid_height = static_cast<int>((max_pt.y() - min_pt.y()) / grid_resolution) + 1;
  
  std::vector<std::vector<GridCell>> grid(grid_height, std::vector<GridCell>(grid_width));
  
  for (size_t i = 0; i < points.size(); ++i)
  {
    int gx = static_cast<int>((points[i].x() - min_pt.x()) / grid_resolution);
    int gy = static_cast<int>((points[i].y() - min_pt.y()) / grid_resolution);
    
    gx = std::max(0, std::min(grid_width - 1, gx));
    gy = std::max(0, std::min(grid_height - 1, gy));
    
    grid[gy][gx].point_indices.push_back(i);
    grid[gy][gx].is_occupied = true;
  }
  
  for (int y = 0; y < grid_height; ++y)
  {
    for (int x = 0; x < grid_width; ++x)
    {
      if (!grid[y][x].is_occupied)
      {
        for (int dy = -1; dy <= 1; ++dy)
        {
          for (int dx = -1; dx <= 1; ++dx)
          {
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < grid_width && ny >= 0 && ny < grid_height && 
                grid[ny][nx].is_occupied)
            {
              grid[y][x].is_occupied = true;
              goto next_cell;
            }
          }
        }
      }
      next_cell:;
    }
  }
  
  float total_area = 0.0f;
  float cell_area = grid_resolution * grid_resolution;
  
  for (int y = 1; y < grid_height - 1; ++y)
  {
    for (int x = 1; x < grid_width - 1; ++x)
    {
      if (grid[y][x].is_occupied)
      {
        bool is_interior = true;
        for (int dy = -1; dy <= 1 && is_interior; ++dy)
        {
          for (int dx = -1; dx <= 1 && is_interior; ++dx)
          {
            if (!grid[y + dy][x + dx].is_occupied)
            {
              is_interior = false;
            }
          }
        }
        
        if (is_interior)
        {
          total_area += cell_area;
        }
      }
    }
  }
  
  return total_area;
}

}



struct LandingUnit::Implementation
{
  LandingConfig config;
  mutable std::mt19937 rng{std::random_device{}()};
  
  mutable size_t total_analyses = 0;
  mutable double total_time_ms = 0.0;
  
  Implementation(const LandingConfig& cfg) : config(cfg) 
  {
  }
  
  PlaneRANSAC fitPlaneRANSAC(const PointCloud& cloud) const
  {
    const size_t n_points = cloud.points.size();
    if (n_points < 3) 
    {
      return {};
    }
    
    PlaneRANSAC best_plane;
    best_plane.score = 0.0f;
    
    std::uniform_int_distribution<size_t> dist(0, n_points - 1);
    
    for (size_t iter = 0; iter < config.ransac_iterations; ++iter)
    {
      size_t i1 = dist(rng);
      size_t i2 = dist(rng);
      size_t i3 = dist(rng);
      
      while (i2 == i1) 
      {
        i2 = dist(rng);
      }
      while (i3 == i1 || i3 == i2) 
      {
        i3 = dist(rng);
      }
      
      Eigen::Vector4f plane = computePlane(cloud.points[i1], 
                                          cloud.points[i2], 
                                          cloud.points[i3]);
      
      std::vector<size_t> inliers;
      for (size_t i = 0; i < n_points; ++i)
      {
        float dist = pointToPlaneDistance(cloud.points[i], plane);
        if (dist <= config.ransac_distance_threshold)
        {
          inliers.push_back(i);
        }
      }
      
      if (inliers.size() > best_plane.inliers.size())
      {
        best_plane.coefficients = plane;
        best_plane.inliers = std::move(inliers);
        best_plane.score = static_cast<float>(best_plane.inliers.size()) / n_points;
      }
    }
    
    if (best_plane.inliers.size() >= config.min_points_for_plane)
    {
      refinePlane(cloud, best_plane);
    }
    
    return best_plane;
  }
  
  void refinePlane(const PointCloud& cloud, PlaneRANSAC& plane) const
  {
    if (plane.inliers.size() < 3) 
    {
      return;
    }
    
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (size_t idx : plane.inliers)
    {
      centroid += cloud.points[idx];
    }
    centroid /= plane.inliers.size();
    
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (size_t idx : plane.inliers)
    {
      Eigen::Vector3f diff = cloud.points[idx] - centroid;
      covariance += diff * diff.transpose();
    }
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    Eigen::Vector3f normal = solver.eigenvectors().col(0);
    
    if (normal.z() < 0) 
    {
      normal = -normal;
    }
    
    float d = -normal.dot(centroid);
    plane.coefficients = Eigen::Vector4f(normal.x(), normal.y(), normal.z(), d);
  }
  
  LandingCandidate evaluateCandidate(const PointCloud& cloud, 
                                    const PlaneRANSAC& plane,
                                    const Eigen::Vector3f& current_position) const
  {
    LandingCandidate candidate;
    candidate.plane_equation = plane.coefficients;
    candidate.normal = plane.coefficients.head<3>().normalized();
    candidate.inlier_indices = plane.inliers;
    
    if (plane.inliers.size() < config.min_points_for_plane)
    {
      return candidate;
    }
    
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    for (size_t idx : plane.inliers)
    {
      centroid += cloud.points[idx];
    }
    centroid /= plane.inliers.size();
    candidate.center = centroid;
    
    Eigen::Vector3f gravity_world(0, 0, -1);
    float cos_angle = std::abs(candidate.normal.dot(-gravity_world));
    candidate.tilt_angle = std::acos(std::min(1.0f, cos_angle)) * RAD_TO_DEG;
    
    Eigen::Vector3f u = candidate.normal.cross(Eigen::Vector3f::UnitX());
    if (u.norm() < 0.1f) 
    {
      u = candidate.normal.cross(Eigen::Vector3f::UnitY());
    }
    u.normalize();
    Eigen::Vector3f v = candidate.normal.cross(u);
    
    std::vector<Eigen::Vector2f> points_2d;
    std::vector<Eigen::Vector3f> projected_3d;
    
    for (size_t idx : plane.inliers)
    {
      Eigen::Vector3f proj = projectOntoPlane(cloud.points[idx], plane.coefficients);
      projected_3d.push_back(proj);
      
      Eigen::Vector3f rel = proj - candidate.center;
      points_2d.emplace_back(rel.dot(u), rel.dot(v));
    }
    
    std::vector<size_t> hull_indices = convexHull2D(points_2d);
    
    std::vector<Eigen::Vector2f> hull_points;
    for (size_t idx : hull_indices)
    {
      hull_points.push_back(points_2d[idx]);
      candidate.boundary_points.push_back(projected_3d[idx]);
    }
    
    float convex_area = computePolygonArea(hull_points);
    float actual_area = computeAreaWithHoles(points_2d, 0.05f);
    
    candidate.area = actual_area;
    
    float concavity_ratio = 1.0f - (actual_area / convex_area);
    if (concavity_ratio > 0.3f)
    {
      candidate.area *= 0.8f;
    }
    
    float sum_sq_dist = 0.0f;
    for (size_t idx : plane.inliers)
    {
      float dist = pointToPlaneDistance(cloud.points[idx], plane.coefficients);
      sum_sq_dist += dist * dist;
    }
    candidate.roughness = std::sqrt(sum_sq_dist / plane.inliers.size());
    
    float distance_to_sensor = (candidate.center - current_position).norm();
    float sensor_angle = std::acos(std::abs(candidate.normal.dot(
        (current_position - candidate.center).normalized())));
    
    float base_density = 10000.0f;
    float distance_factor = 1.0f / (1.0f + distance_to_sensor * 0.1f);
    float angle_factor = std::cos(sensor_angle);
    
    float expected_points = candidate.area * base_density * distance_factor * angle_factor;
    candidate.confidence = std::min(1.0f, plane.inliers.size() / expected_points);
    
    float tilt_score = 1.0f - std::min(1.0f, candidate.tilt_angle / config.max_tilt_angle);
    float area_score = std::min(1.0f, candidate.area / (config.min_area * 4.0f));
    float roughness_score = 1.0f - std::min(1.0f, candidate.roughness / config.max_roughness);
    
    candidate.score = 0.33f * tilt_score + 0.33f * area_score + 0.33f * roughness_score;
    
    return candidate;
  }
  
  LandingResult analyze(const PointCloud& cloud, 
                       const Eigen::Vector3f& current_position)
  {
    auto start_time = std::chrono::steady_clock::now();
    
    LandingResult result;
    
    PointCloud remaining_cloud = cloud;
    
    while (remaining_cloud.points.size() >= config.min_points_for_plane && 
           result.candidates.size() < 5)
    {
      PlaneRANSAC plane = fitPlaneRANSAC(remaining_cloud);
      
      if (plane.inliers.size() < config.min_points_for_plane)
      {
        break;
      }
      
      LandingCandidate candidate = evaluateCandidate(cloud, plane, current_position);
      
      if (candidate.area >= config.min_area &&
          candidate.tilt_angle <= config.max_tilt_angle &&
          candidate.roughness <= config.max_roughness)
      {
        result.candidates.push_back(candidate);
      }
      
      std::vector<Eigen::Vector3f> new_points;
      std::set<size_t> inlier_set(plane.inliers.begin(), plane.inliers.end());
      
      for (size_t i = 0; i < remaining_cloud.points.size(); ++i)
      {
        if (inlier_set.find(i) == inlier_set.end())
        {
          new_points.push_back(remaining_cloud.points[i]);
        }
      }
      
      remaining_cloud.points = std::move(new_points);
    }
    
    if (!result.candidates.empty())
    {
      auto best_it = std::max_element(result.candidates.begin(), 
                                     result.candidates.end(),
                                     [](const auto& a, const auto& b) 
                                     {
                                       return a.score < b.score;
                                     });
      
      result.best_candidate = *best_it;
      result.has_valid_landing_zone = true;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.computation_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();
    
    total_analyses++;
    total_time_ms += result.computation_time_ms;
    
    return result;
  }
  
  nlohmann::json toJSON() const
  {
    nlohmann::json j;
    
    j["config"]["ransac_iterations"] = config.ransac_iterations;
    j["config"]["ransac_distance_threshold"] = config.ransac_distance_threshold;
    j["config"]["min_points_for_plane"] = config.min_points_for_plane;
    j["config"]["max_tilt_angle"] = config.max_tilt_angle;
    j["config"]["min_area"] = config.min_area;
    j["config"]["max_roughness"] = config.max_roughness;
    
    j["statistics"]["total_analyses"] = total_analyses;
    j["statistics"]["average_time_ms"] = total_analyses > 0 ? 
        total_time_ms / total_analyses : 0.0;
    
    return j;
  }
};

LandingUnit::LandingUnit(const LandingConfig& config)
  : impl_(std::make_unique<Implementation>(config))
{
  spdlog::info("LandingUnit initialized with min_area={:.2f}m², max_tilt={:.1f}°", 
               config.min_area, config.max_tilt_angle);
}

LandingUnit::~LandingUnit() = default;

LandingResult LandingUnit::analyzePointCloud(const PointCloud& cloud,
                                            const Eigen::Vector3f& current_position)
{
  return impl_->analyze(cloud, current_position);
}

void LandingUnit::updateConfig(const LandingConfig& config)
{
  impl_->config = config;
  spdlog::info("LandingUnit config updated");
}

const LandingConfig& LandingUnit::getConfig() const
{
  return impl_->config;
}

void LandingUnit::exportToJSON(const std::string& filename) const
{
  std::ofstream file(filename);
  if (file.is_open())
  {
    file << impl_->toJSON().dump(2);
    spdlog::info("LandingUnit exported to {}", filename);
  }
  else
  {
    spdlog::error("Failed to export LandingUnit to {}", filename);
  }
}

}
}