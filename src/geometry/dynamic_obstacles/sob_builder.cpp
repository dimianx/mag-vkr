#include "vkr/geometry/dynamic_obstacles/sob_builder.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <Eigen/Dense>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

void buildBVHFromBuilder(SweptOBBBVH* bvh, std::vector<SweptOBB>&& obbs);

namespace
{

bool loadOBJVertices(const std::string& filename, std::vector<Eigen::Vector3f>& output_vertices)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    spdlog::error("SweptOBBBVHBuilder: Failed to open OBJ file: {}", filename);
    return false;
  }
  
  std::vector<Eigen::Vector3f> vertices;
  output_vertices.clear();
  
  std::string line;
  while (std::getline(file, line))
  {
    if (line.empty() || line[0] == '#')
    {
      continue;
    }
    
    std::istringstream iss(line);
    std::string prefix;
    iss >> prefix;
    
    if (prefix == "v")
    {
      float x, y, z;
      iss >> x >> y >> z;
      vertices.emplace_back(x, y, z);
    }
    else if (prefix == "f")
    {
      std::vector<int> indices;
      std::string vertex_str;
      
      while (iss >> vertex_str)
      {
        size_t slash_pos = vertex_str.find('/');
        std::string v_idx_str = (slash_pos != std::string::npos) ?
                               vertex_str.substr(0, slash_pos) : vertex_str;
        
        int v_idx = std::stoi(v_idx_str);
        
        if (v_idx < 0)
        {
          v_idx = static_cast<int>(vertices.size()) + v_idx + 1;
        }
        indices.push_back(v_idx - 1);
      }
      
      for (int idx : indices)
      {
        if (idx >= 0 && static_cast<size_t>(idx) < vertices.size())
        {
          output_vertices.push_back(vertices[idx]);
        }
      }
    }
  }
  
  spdlog::info("SweptOBBBVHBuilder: Loaded {} vertices from OBJ file", output_vertices.size());
  
  return !output_vertices.empty();
}

}

OBB computeOptimalOBB(const std::vector<Eigen::Vector3f>& points)
{
  OBB result;
  
  if (points.empty())
  {
    result.center = Eigen::Vector3f::Zero();
    result.half_extents = Eigen::Vector3f::Ones() * 0.1f;
    result.orientation = Eigen::Matrix3f::Identity();
    return result;
  }
  
  if (points.size() == 1)
  {
    result.center = points[0];
    result.half_extents = Eigen::Vector3f::Ones() * 0.01f;
    result.orientation = Eigen::Matrix3f::Identity();
    return result;
  }
  
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (const auto& p : points)
  {
    mean += p;
  }
  mean /= static_cast<float>(points.size());
  result.center = mean;
  
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (const auto& p : points)
  {
    Eigen::Vector3f diff = p - mean;
    cov += diff * diff.transpose();
  }
  cov /= static_cast<float>(points.size());
  
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
  if (eigensolver.info() != Eigen::Success)
  {
    Eigen::Vector3f min_pt = points[0];
    Eigen::Vector3f max_pt = points[0];
    for (const auto& p : points)
    {
      min_pt = min_pt.cwiseMin(p);
      max_pt = max_pt.cwiseMax(p);
    }
    
    result.center = (min_pt + max_pt) * 0.5f;
    result.half_extents = (max_pt - min_pt) * 0.5f;
    result.orientation = Eigen::Matrix3f::Identity();
    return result;
  }
  
  result.orientation = eigensolver.eigenvectors();
  
  Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
  const float epsilon = 1e-6f;
  
  if (result.orientation.determinant() < 0)
  {
    result.orientation.col(0) = -result.orientation.col(0);
  }
  
  Eigen::Vector3f min_proj = Eigen::Vector3f::Constant(std::numeric_limits<float>::max());
  Eigen::Vector3f max_proj = Eigen::Vector3f::Constant(std::numeric_limits<float>::lowest());
  
  for (const auto& p : points)
  {
    Eigen::Vector3f local_p = result.orientation.transpose() * (p - mean);
    min_proj = min_proj.cwiseMin(local_p);
    max_proj = max_proj.cwiseMax(local_p);
  }
  
  result.half_extents = (max_proj - min_proj) * 0.5f;
  
  for (int i = 0; i < 3; ++i)
  {
    if (eigenvalues[i] < epsilon * eigenvalues.maxCoeff())
    {
      result.half_extents[i] = std::max(result.half_extents[i], GEOM_EPS);
    }
    else
    {
      result.half_extents[i] = std::max(result.half_extents[i], 0.01f);
    }
  }
  
  Eigen::Vector3f center_offset = (max_proj + min_proj) * 0.5f;
  result.center += result.orientation * center_offset;
  
  return result;
}

struct SweptOBBBVHBuilder::Implementation 
{
  std::vector<SweptOBB> swept_obbs;
  int trajectory_degree = 5;
  GeometryConfig config;
};

SweptOBBBVHBuilder::SweptOBBBVHBuilder(const GeometryConfig& config)
    : impl_(std::make_unique<Implementation>()),
      config_(config)
{
  impl_->config = config;
}

SweptOBBBVHBuilder::~SweptOBBBVHBuilder() = default;

SweptOBBBVHBuilder& SweptOBBBVHBuilder::addFromMesh(
    const std::string& filename,
    ObjectId object_id,
    const std::vector<Eigen::Vector3f>& position_trajectory,
    const std::vector<Eigen::Vector3f>& rotation_trajectory,
    const std::chrono::steady_clock::time_point& t_start,
    const std::chrono::steady_clock::time_point& t_end)
{
  std::vector<Eigen::Vector3f> vertices;
  
  if (!loadOBJVertices(filename, vertices)) 
  {
    spdlog::warn("Failed to load mesh from {}", filename);
    return *this;
  }
  
  OBB obb = computeOptimalOBB(vertices);
  
  SweptOBB swept;
  swept.base_obb = obb;
  swept.object_id = object_id;
  swept.position_coeffs = position_trajectory;
  swept.rotation_coeffs = rotation_trajectory;
  swept.t0 = t_start;
  swept.t1 = t_end;
  
  swept.computeBounds();
  
  impl_->swept_obbs.push_back(swept);
  
  spdlog::debug("Added swept OBB from mesh {} with {} vertices", 
                filename, vertices.size());
  
  return *this;
}

SweptOBBBVHBuilder& SweptOBBBVHBuilder::addWithTimeMapping(
    const OBB& obb,
    ObjectId object_id,
    const std::vector<Eigen::Vector3f>& position_trajectory,
    const std::vector<Eigen::Vector3f>& rotation_trajectory,
    const std::vector<float>& time_coeffs,
    const std::chrono::steady_clock::time_point& t_start,
    const std::chrono::steady_clock::time_point& t_end)
{
  SweptOBB swept;
  swept.base_obb = obb;
  swept.object_id = object_id;
  swept.position_coeffs = position_trajectory;
  swept.rotation_coeffs = rotation_trajectory;
  swept.time_coeffs = time_coeffs;
  swept.t0 = t_start;
  swept.t1 = t_end;
  
  swept.computeBounds();
  
  impl_->swept_obbs.push_back(swept);
  
  return *this;
}

SweptOBBBVHBuilder& SweptOBBBVHBuilder::setTrajectoryDegree(int degree)
{
  impl_->trajectory_degree = std::clamp(degree, 1, 7);
  return *this;
}

std::unique_ptr<SweptOBBBVH> SweptOBBBVHBuilder::build()
{
  size_t obb_count = impl_->swept_obbs.size();
  
  auto bvh = std::unique_ptr<SweptOBBBVH>(new SweptOBBBVH(config_));
  
  buildBVHFromBuilder(bvh.get(), std::move(impl_->swept_obbs));
  
  spdlog::info("Built BVH with {} swept OBBs", obb_count);
  
  impl_->swept_obbs.clear();
  
  return bvh;
}

SweptOBBBVHBuilder& SweptOBBBVHBuilder::clear()
{
  impl_->swept_obbs.clear();
  return *this;
}

size_t SweptOBBBVHBuilder::getObstacleCount() const
{
  return impl_->swept_obbs.size();
}

}
}
}