#include "vkr/geometry/static_obstacles/bst_builder.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <Eigen/Geometry> 
#include <Eigen/Dense>  

namespace vkr
{
namespace geometry
{
namespace static_obstacles
{

struct BSTBuilder::Implementation
{
  GeometryConfig config;
  std::vector<BoundingSphereTree::Face> faces;
  
  explicit Implementation(const GeometryConfig& cfg) : config(cfg)
  {
  }
  
  bool loadOBJ(const std::string& filename, ObjectId object_id);
  void computeFaceNormal(BoundingSphereTree::Face& face);
};

BSTBuilder::BSTBuilder(const GeometryConfig& config)
  : impl_(std::make_unique<Implementation>(config))
{
}

BSTBuilder::~BSTBuilder() = default;

bool BSTBuilder::loadMesh(const std::string& filename, ObjectId object_id)
{
  size_t dot_pos = filename.find_last_of('.');
  if (dot_pos == std::string::npos) 
  {
    spdlog::error("BSTBuilder: No file extension found in {}", filename);
    return false;
  }
  
  std::string extension = filename.substr(dot_pos + 1);
  std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
  
  if (extension == "obj") 
  {
    return impl_->loadOBJ(filename, object_id);
  } 
  else 
  {
    spdlog::error("BSTBuilder: Unsupported file format .{}", extension);
    return false;
  }
}

void BSTBuilder::addFace(const Eigen::Vector3f& v0,
                        const Eigen::Vector3f& v1,
                        const Eigen::Vector3f& v2,
                        ObjectId object_id)
{
  BoundingSphereTree::Face face;
  face.vertices[0] = v0;
  face.vertices[1] = v1;
  face.vertices[2] = v2;
  face.object_id = object_id;
  
  impl_->computeFaceNormal(face);
  impl_->faces.push_back(face);
}

void BSTBuilder::clear()
{
  impl_->faces.clear();
}

std::unique_ptr<BoundingSphereTree> BSTBuilder::build()
{
  if (impl_->faces.empty()) 
  {
    spdlog::warn("BSTBuilder: No faces to build tree from");
    return nullptr;
  }
  
  auto tree = std::make_unique<BoundingSphereTree>();
  tree->build(impl_->faces, impl_->config.max_faces_per_leaf);
  
  spdlog::info("BSTBuilder: Built tree with {} faces", impl_->faces.size());
  
  return tree;
}

size_t BSTBuilder::getFaceCount() const
{
  return impl_->faces.size();
}

void BSTBuilder::exportToJSON(const std::string& filename) const
{
  nlohmann::json j;
  
  j["face_count"] = impl_->faces.size();
  
  nlohmann::json faces_json = nlohmann::json::array();
  
  size_t max_faces_to_export = 10000;
  size_t step = impl_->faces.size() > max_faces_to_export ? 
                impl_->faces.size() / max_faces_to_export : 1;
  
  for (size_t i = 0; i < impl_->faces.size(); i += step) 
  {
    const auto& face = impl_->faces[i];
    nlohmann::json face_json;
    
    face_json["index"] = i;
    face_json["vertices"] = nlohmann::json::array();
    for (int v = 0; v < 3; ++v) 
    {
      face_json["vertices"].push_back(
      {
        face.vertices[v].x(),
        face.vertices[v].y(),
        face.vertices[v].z()
      });
    }
    face_json["normal"] = 
    {
      face.normal.x(), face.normal.y(), face.normal.z()
    };
    face_json["object_id"] = face.object_id;
    
    faces_json.push_back(face_json);
  }
  
  j["faces_sample"] = faces_json;
  j["total_faces"] = impl_->faces.size();
  
  if (!impl_->faces.empty()) 
  {
    Eigen::Vector3f min_pt = impl_->faces[0].vertices[0];
    Eigen::Vector3f max_pt = impl_->faces[0].vertices[0];
    
    for (const auto& face : impl_->faces) 
    {
      for (int i = 0; i < 3; ++i) 
      {
        min_pt = min_pt.cwiseMin(face.vertices[i]);
        max_pt = max_pt.cwiseMax(face.vertices[i]);
      }
    }
    
    j["bounding_box"] = 
    {
      {
        "min", 
        {
          min_pt.x(), min_pt.y(), min_pt.z()
        }
      },
      {
        "max", 
        {
          max_pt.x(), max_pt.y(), max_pt.z()
        }
      },
      {
        "size", 
        {
          (max_pt - min_pt).x(), (max_pt - min_pt).y(), (max_pt - min_pt).z()
        }
      }
    };
  }
  
  std::ofstream file(filename);
  file << j.dump(2);
}

void BSTBuilder::Implementation::computeFaceNormal(BoundingSphereTree::Face& face)
{
  Eigen::Vector3f e1 = face.vertices[1] - face.vertices[0];
  Eigen::Vector3f e2 = face.vertices[2] - face.vertices[0];
  face.normal = e1.cross(e2).normalized();
}

bool BSTBuilder::Implementation::loadOBJ(const std::string& filename, ObjectId object_id)
{
  std::ifstream file(filename);
  if (!file.is_open()) 
  {
    spdlog::error("BSTBuilder: Failed to open OBJ file: {}", filename);
    return false;
  }
  
  std::vector<Eigen::Vector3f> vertices;
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
      
      if (indices.size() >= 3) 
      {
        for (size_t i = 1; i < indices.size() - 1; ++i) 
        {
          if (indices[0] >= 0 && static_cast<size_t>(indices[0]) < vertices.size() && 
              indices[i] >= 0 && static_cast<size_t>(indices[i]) < vertices.size() && 
              indices[i+1] >= 0 && static_cast<size_t>(indices[i+1]) < vertices.size()) 
          {
            BoundingSphereTree::Face face;
            face.vertices[0] = vertices[indices[0]];
            face.vertices[1] = vertices[indices[i]];
            face.vertices[2] = vertices[indices[i+1]];
            face.object_id = object_id;
            
            computeFaceNormal(face);
            faces.push_back(face);
          }
        }
      }
    }
  }
  
  spdlog::info("BSTBuilder: Loaded {} vertices and {} faces from OBJ file", 
               vertices.size(), faces.size());
  
  return !faces.empty();
}

}
}
}