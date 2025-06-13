#include "vkr/geometry/dynamic_obstacles/swept_obb_bvh.hpp"
#include "vkr/geometry/dynamic_obstacles/sob_builder.hpp"
#include "vkr/geometry/intersections.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <stack>
#include <chrono>
#include <fstream>

namespace vkr
{
namespace geometry
{
namespace dynamic_obstacles
{

struct SweptOBBBVH::BVHNode 
{
  OBB bounding_obb;
  BoundingBox spatial_aabb;
  std::chrono::steady_clock::time_point t_min;
  std::chrono::steady_clock::time_point t_max;
  
  std::unique_ptr<BVHNode> left;
  std::unique_ptr<BVHNode> right;
  
  std::vector<SweptOBB*> obbs;
  
  bool isLeaf() const { return !left && !right; }
};

struct SweptOBBBVH::Implementation 
{
  GeometryConfig config;
  std::unique_ptr<BVHNode> root;
  std::vector<SweptOBB> all_obbs;
  
  mutable Stats stats;
  
  void buildBVH(std::vector<SweptOBB>&& obbs);
  std::unique_ptr<BVHNode> buildNode(std::vector<SweptOBB*>& obbs, int depth);
  
  void queryNode(const BVHNode* node,
                const OBB& query,
                const std::chrono::steady_clock::time_point& t0,
                const std::chrono::steady_clock::time_point& t1,
                collision::HitResult& result) const;
  
  OBB computeBoundingOBB(const std::vector<SweptOBB*>& obbs) const;
};

SweptOBBBVH::SweptOBBBVH(const GeometryConfig& config)
    : impl_(std::make_unique<Implementation>())
{
  impl_->config = config;
}

SweptOBBBVH::~SweptOBBBVH() = default;

collision::HitResult SweptOBBBVH::queryPoint(
    const Eigen::Vector3f& point,
    const std::chrono::steady_clock::time_point& time) const
{
  OBB point_obb;
  point_obb.center = point;
  point_obb.half_extents = Eigen::Vector3f::Zero();
  point_obb.orientation = Eigen::Matrix3f::Identity();
  
  return queryOBB(point_obb, time, time);
}

collision::HitResult SweptOBBBVH::queryOBB(
    const OBB& obb,
    const std::chrono::steady_clock::time_point& t0,
    const std::chrono::steady_clock::time_point& t1) const
{
  collision::HitResult result;
  result.hit = false;
  result.t_min = INFINITY_F;
  result.object_id = INVALID_ID;
  result.hit_point = Eigen::Vector3f::Zero();
  result.hit_normal = Eigen::Vector3f::Zero();
  result.object_type = collision::HitResult::ObjectType::NONE;
  
  if (!impl_->root) 
  {
    return result;
  }
  
  impl_->queryNode(impl_->root.get(), obb, t0, t1, result);
  
  return result;
}

collision::HitResult SweptOBBBVH::queryPath(
    const std::vector<Eigen::Vector3f>& waypoints,
    const Eigen::Vector3f& half_extents,
    const std::chrono::steady_clock::time_point& start_time,
    float velocity) const
{
  collision::HitResult result;
  result.hit = false;
  result.t_min = INFINITY_F;
  result.object_id = INVALID_ID;
  result.hit_point = Eigen::Vector3f::Zero();
  result.hit_normal = Eigen::Vector3f::Zero();
  result.object_type = collision::HitResult::ObjectType::NONE;
  
  if (waypoints.size() < 2) 
  {
    return result;
  }
  
  auto current_time = start_time;
  
  for (size_t i = 0; i < waypoints.size() - 1; ++i) 
  {
    OBB segment_obb;
    segment_obb.half_extents = half_extents;
    segment_obb.orientation = Eigen::Matrix3f::Identity();
    
    float length = (waypoints[i + 1] - waypoints[i]).norm();
    auto duration = std::chrono::milliseconds(static_cast<long>(length / velocity * 1000));
    
    const int samples = std::max(2, static_cast<int>(length / (half_extents.maxCoeff() * 2)));
    for (int j = 0; j < samples; ++j) 
    {
      float t = static_cast<float>(j) / (samples - 1);
      segment_obb.center = waypoints[i] * (1 - t) + waypoints[i + 1] * t;
      
      auto sample_time = current_time + std::chrono::milliseconds(
          static_cast<long>(t * duration.count()));
      
      auto segment_result = queryOBB(segment_obb, sample_time, sample_time);
      
      if (segment_result.hit) 
      {
        return segment_result;
      }
    }
    
    current_time += duration;
  }
  
  return result;
}

void SweptOBBBVH::batchQueryPoints(
    const Eigen::Vector3f* points,
    size_t count,
    const std::chrono::steady_clock::time_point& time,
    collision::HitResult* results) const
{
  namespace simd = math::simd;
  
  simd::simdLoop(count, [&](size_t i, bool) 
  {
    results[i] = queryPoint(points[i], time);
  });
}

std::vector<OBB> SweptOBBBVH::getActiveOBBs(
    const std::chrono::steady_clock::time_point& time) const
{
  std::vector<OBB> result;
  
  for (const auto& swept : impl_->all_obbs) 
  {
    if (swept.isActiveAtTime(time)) 
    {
      result.push_back(swept.getOBBAtTime(time));
    }
  }
  
  return result;
}

float SweptOBBBVH::getDistance(
    const Eigen::Vector3f& point,
    const std::chrono::steady_clock::time_point& time) const
{
  float min_distance = INFINITY_F;
  
  for (const auto& swept : impl_->all_obbs) 
  {
    if (swept.isActiveAtTime(time)) 
    {
      float dist = swept.distanceToPoint(point, time);
      min_distance = std::min(min_distance, dist);
    }
  }
  
  return min_distance;
}

SweptOBBBVH::Stats SweptOBBBVH::getStats() const
{
  return impl_->stats;
}

void SweptOBBBVH::exportToJSON(const std::string& filename) const
{
  nlohmann::json j;
  
  j["stats"] = {
      {"num_objects", impl_->stats.num_objects},
      {"num_nodes", impl_->stats.num_nodes},
      {"tree_depth", impl_->stats.tree_depth},
      {"num_leaves", impl_->stats.num_leaves},
      {"avg_obbs_per_leaf", impl_->stats.avg_obbs_per_leaf}
  };
  
  std::function<nlohmann::json(const BVHNode*)> exportNode = 
      [&](const BVHNode* node) -> nlohmann::json 
  {
    if (!node) return nullptr;
    
    nlohmann::json n;
    n["is_leaf"] = node->isLeaf();
    
    if (node->isLeaf()) 
    {
      n["num_obbs"] = node->obbs.size();
    }
    else 
    {
      n["left"] = exportNode(node->left.get());
      n["right"] = exportNode(node->right.get());
    }
    
    n["spatial_aabb"] = {
        {"min", {node->spatial_aabb.min.x(), node->spatial_aabb.min.y(), node->spatial_aabb.min.z()}},
        {"max", {node->spatial_aabb.max.x(), node->spatial_aabb.max.y(), node->spatial_aabb.max.z()}}
    };
    
    return n;
  };
  
  j["tree"] = exportNode(impl_->root.get());
  
  std::ofstream file(filename);
  file << j.dump(2);
}

void SweptOBBBVH::Implementation::buildBVH(std::vector<SweptOBB>&& obbs)
{
  auto start_time = std::chrono::high_resolution_clock::now();
  
  all_obbs = std::move(obbs);
  
  if (all_obbs.empty())
  {
    root = nullptr;
    
    stats.num_objects = 0;
    stats.num_nodes = 0;
    stats.tree_depth = 0;
    stats.num_leaves = 0;
    stats.avg_obbs_per_leaf = 0.0f;
    stats.construction_time_ms = 0.0f;
    
    spdlog::debug("Built empty BVH");
    return;
  }
  
  for (auto& obb : all_obbs) 
  {
    obb.computeBounds();
  }
  
  std::vector<SweptOBB*> ptrs;
  ptrs.reserve(all_obbs.size());
  for (auto& obb : all_obbs) 
  {
    ptrs.push_back(&obb);
  }
  
  root = buildNode(ptrs, 0);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  stats.construction_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
  
  stats.num_objects = all_obbs.size();
  stats.num_nodes = 0;
  stats.tree_depth = 0;
  stats.num_leaves = 0;
  stats.avg_obbs_per_leaf = 0.0f;
  
  if (root)
  {
    std::function<void(const BVHNode*, int)> computeStats = 
        [&](const BVHNode* node, int depth) 
    {
      if (!node) return;
      
      stats.num_nodes++;
      stats.tree_depth = std::max(stats.tree_depth, static_cast<size_t>(depth));
      
      if (node->isLeaf()) 
      {
        stats.num_leaves++;
        stats.avg_obbs_per_leaf += node->obbs.size();
      }
      else 
      {
        computeStats(node->left.get(), depth + 1);
        computeStats(node->right.get(), depth + 1);
      }
    };
    
    computeStats(root.get(), 1);
    
    if (stats.num_leaves > 0) 
    {
      stats.avg_obbs_per_leaf /= stats.num_leaves;
    }
  }
  
  spdlog::debug("Built BVH with {} nodes, depth {}, {} leaves, {:.1f} OBBs/leaf in {:.2f}ms",
                stats.num_nodes, stats.tree_depth, stats.num_leaves,
                stats.avg_obbs_per_leaf, stats.construction_time_ms);
}

std::unique_ptr<SweptOBBBVH::BVHNode> 
SweptOBBBVH::Implementation::buildNode(std::vector<SweptOBB*>& obbs, int depth)
{
  if (obbs.empty())
  {
    return nullptr;
  }
  
  auto node = std::make_unique<BVHNode>();
  
  bool first = true;
  for (auto* obb : obbs) 
  {
    if (first) 
    {
      node->spatial_aabb = obb->cached.spatial_aabb;
      node->bounding_obb = obb->cached.bounding_obb;
      node->t_min = obb->t0;
      node->t_max = obb->isRepeating() ? 
                    std::chrono::steady_clock::time_point::max() : obb->t1;
      first = false;
    }
    else 
    {
      node->spatial_aabb.expand(obb->cached.spatial_aabb.min);
      node->spatial_aabb.expand(obb->cached.spatial_aabb.max);
      node->t_min = std::min(node->t_min, obb->t0);
      if (!obb->isRepeating()) 
      {
        node->t_max = std::max(node->t_max, obb->t1);
      }
      else 
      {
        node->t_max = std::chrono::steady_clock::time_point::max();
      }
    }
  }
  
  node->bounding_obb = computeBoundingOBB(obbs);
  
  if (obbs.size() <= config.bvh_max_leaf_size || depth > 20) 
  {
    node->obbs = obbs;
    return node;
  }
  
  Eigen::Vector3f min_center = obbs[0]->cached.bounding_obb.center;
  Eigen::Vector3f max_center = obbs[0]->cached.bounding_obb.center;
  
  for (auto* obb : obbs)
  {
    min_center = min_center.cwiseMin(obb->cached.bounding_obb.center);
    max_center = max_center.cwiseMax(obb->cached.bounding_obb.center);
  }
  
  Eigen::Vector3f extent = max_center - min_center;
  
  float time_extent = 0.0f;
  if (obbs.size() > 1)
  {
    auto min_time = std::min_element(obbs.begin(), obbs.end(), 
        [](const SweptOBB* a, const SweptOBB* b)
        {
          auto a_center = a->t0 + (a->t1 - a->t0) / 2;
          auto b_center = b->t0 + (b->t1 - b->t0) / 2;
          return a_center < b_center;
        });
    auto max_time = std::max_element(obbs.begin(), obbs.end(), 
        [](const SweptOBB* a, const SweptOBB* b)
        {
          auto a_center = a->t0 + (a->t1 - a->t0) / 2;
          auto b_center = b->t0 + (b->t1 - b->t0) / 2;
          return a_center < b_center;
        });
    
    auto t_min = (*min_time)->t0 + ((*min_time)->t1 - (*min_time)->t0) / 2;
    auto t_max = (*max_time)->t0 + ((*max_time)->t1 - (*max_time)->t0) / 2;
    time_extent = std::chrono::duration<float>(t_max - t_min).count();
  }
  
  int split_axis = 0;
  float max_extent = extent[0];
  
  if (extent[1] > max_extent)
  {
    split_axis = 1;
    max_extent = extent[1];
  }
  if (extent[2] > max_extent)
  {
    split_axis = 2;
    max_extent = extent[2];
  }
  
  float normalized_time_extent = time_extent * 10.0f;
  if (config.bvh_use_temporal_splits && normalized_time_extent > max_extent)
  {
    split_axis = 3;
  }
  
  if (split_axis < 3)
  {
    std::sort(obbs.begin(), obbs.end(), 
        [split_axis](const SweptOBB* a, const SweptOBB* b)
        {
          return a->cached.bounding_obb.center[split_axis] < 
                 b->cached.bounding_obb.center[split_axis];
        });
  }
  else
  {
    std::sort(obbs.begin(), obbs.end(), 
        [](const SweptOBB* a, const SweptOBB* b)
        {
          auto a_center = a->t0 + (a->t1 - a->t0) / 2;
          auto b_center = b->t0 + (b->t1 - b->t0) / 2;
          return a_center < b_center;
        });
  }
  
  size_t mid = obbs.size() / 2;
  std::vector<SweptOBB*> left_obbs(obbs.begin(), obbs.begin() + mid);
  std::vector<SweptOBB*> right_obbs(obbs.begin() + mid, obbs.end());
  
  node->left = buildNode(left_obbs, depth + 1);
  node->right = buildNode(right_obbs, depth + 1);
  
  return node;
}

void SweptOBBBVH::Implementation::queryNode(
    const BVHNode* node,
    const OBB& query,
    const std::chrono::steady_clock::time_point& t0,
    const std::chrono::steady_clock::time_point& t1,
    collision::HitResult& result) const
{
  if (!node) return;
  
  if (node->t_max < t0 || node->t_min > t1) 
  {
    return;
  }
  
  BoundingBox query_aabb = query.getAABB();
  
  if (!node->spatial_aabb.intersects(query_aabb)) 
  {
    return;
  }
  
  if (!intersectOBBOBB(query, node->bounding_obb)) 
  {
    return;
  }
  
  if (node->isLeaf()) 
  {
    for (SweptOBB* swept : node->obbs) 
    {
      if (swept->t1 < t0 || swept->t0 > t1) 
      {
        continue;
      }
      
      auto query_start = std::max(swept->t0, t0);
      auto query_end = std::min(swept->t1, t1);
      
      const int num_samples = 20;
      for (int i = 0; i <= num_samples; ++i) 
      {
        float t = static_cast<float>(i) / num_samples;
        auto sample_time = query_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<float>(t * 
            std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count() / 1000.0f));
        
        if (!swept->isActiveAtTime(sample_time)) continue;
        
        OBB swept_obb = swept->getOBBAtTime(sample_time);
        
        if (intersectOBBOBB(query, swept_obb)) 
        {
          result.hit = true;
          result.object_id = swept->object_id;
          result.object_type = collision::HitResult::ObjectType::DYNAMIC_OBSTACLE;
          
          result.hit_point = (query.center + swept_obb.center) * 0.5f;
          
          result.hit_normal = (query.center - swept_obb.center).normalized();
          
          return;
        }
      }
    }
  }
  else 
  {
    queryNode(node->left.get(), query, t0, t1, result);
    if (result.hit) return;
    
    queryNode(node->right.get(), query, t0, t1, result);
  }
}

OBB SweptOBBBVH::Implementation::computeBoundingOBB(
    const std::vector<SweptOBB*>& obbs) const
{
  if (obbs.empty()) 
  {
    OBB default_obb;
    default_obb.center = Eigen::Vector3f::Zero();
    default_obb.half_extents = Eigen::Vector3f::Ones() * 0.1f;
    default_obb.orientation = Eigen::Matrix3f::Identity();
    return default_obb;
  }
  
  std::vector<Eigen::Vector3f> points;
  for (auto* obb : obbs) 
  {
    auto corners = obb->cached.bounding_obb.getCorners();
    points.insert(points.end(), corners.begin(), corners.end());
  }
  
  return computeOptimalOBB(points);
}

void buildBVHFromBuilder(SweptOBBBVH* bvh, std::vector<SweptOBB>&& obbs)
{
  bvh->impl_->buildBVH(std::move(obbs));
}

}
}
}