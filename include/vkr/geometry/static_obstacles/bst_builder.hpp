#ifndef VKR_GEOMETRY_STATIC_OBSTACLES_BST_BUILDER_HPP_
#define VKR_GEOMETRY_STATIC_OBSTACLES_BST_BUILDER_HPP_

#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/config.hpp"
#include <string>
#include <memory>

namespace vkr
{
namespace geometry
{
namespace static_obstacles
{

class BSTBuilder
{
public:
  explicit BSTBuilder(const GeometryConfig& config);
  ~BSTBuilder();
  
  bool loadMesh(const std::string& filename, ObjectId object_id);
  
  void addFace(const Eigen::Vector3f& v0,
               const Eigen::Vector3f& v1,
               const Eigen::Vector3f& v2,
               ObjectId object_id);
  
  void clear();
  
  std::unique_ptr<BoundingSphereTree> build();
  
  size_t getFaceCount() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

}
}
}

#endif