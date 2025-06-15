#include <gtest/gtest.h>
#include "vkr/geometry/static_obstacles/bounding_sphere_tree.hpp"
#include "vkr/geometry/static_obstacles/bst_builder.hpp"
#include "vkr/geometry/collision/hit_result.hpp"
#include "vkr/config.hpp"
#include <cmath>
#include <random>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace vkr 
{
namespace geometry 
{
namespace static_obstacles 
{

class BoundingSphereTreeTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    createCubeFaces();
    
    createSphereFaces();
    
    createSingleTriangle();
  }
  
  void createCubeFaces() 
  {
    std::vector<Eigen::Vector3f> vertices = 
    {
      {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
      {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}
    };
    
    std::vector<std::array<int, 3>> indices = 
    {
      {0, 3, 2}, {0, 2, 1},
      {4, 5, 6}, {4, 6, 7},
      {0, 4, 7}, {0, 7, 3},
      {1, 2, 6}, {1, 6, 5},
      {0, 1, 5}, {0, 5, 4},
      {3, 7, 6}, {3, 6, 2}
    };
    
    cube_faces_.clear();
    for (const auto& tri : indices) 
    {
      BoundingSphereTree::Face face;
      face.vertices[0] = vertices[tri[0]];
      face.vertices[1] = vertices[tri[1]];
      face.vertices[2] = vertices[tri[2]];
      face.object_id = 1;
      
      Eigen::Vector3f e1 = face.vertices[1] - face.vertices[0];
      Eigen::Vector3f e2 = face.vertices[2] - face.vertices[0];
      face.normal = e1.cross(e2).normalized();
      
      cube_faces_.push_back(face);
    }
  }
  
  void createSphereFaces() 
  {
    sphere_faces_.clear();
    const int latitude_segments = 10;
    const int longitude_segments = 10;
    const float radius = 2.0f;
    
    std::vector<Eigen::Vector3f> vertices;
    
    for (int lat = 0; lat <= latitude_segments; ++lat) 
    {
      float theta = lat * M_PI / latitude_segments;
      float sin_theta = std::sin(theta);
      float cos_theta = std::cos(theta);
      
      for (int lon = 0; lon <= longitude_segments; ++lon) 
      {
        float phi = lon * 2 * M_PI / longitude_segments;
        float sin_phi = std::sin(phi);
        float cos_phi = std::cos(phi);
        
        float x = cos_phi * sin_theta;
        float y = cos_theta;
        float z = sin_phi * sin_theta;
        
        vertices.emplace_back(radius * x, radius * y, radius * z);
      }
    }
    
    for (int lat = 0; lat < latitude_segments; ++lat) 
    {
      for (int lon = 0; lon < longitude_segments; ++lon) 
      {
        int first = lat * (longitude_segments + 1) + lon;
        int second = first + longitude_segments + 1;
        
        BoundingSphereTree::Face face1;
        face1.vertices[0] = vertices[first];
        face1.vertices[1] = vertices[second];
        face1.vertices[2] = vertices[first + 1];
        face1.object_id = 2;
        
        Eigen::Vector3f e1 = face1.vertices[1] - face1.vertices[0];
        Eigen::Vector3f e2 = face1.vertices[2] - face1.vertices[0];
        face1.normal = e1.cross(e2).normalized();
        sphere_faces_.push_back(face1);
        
        BoundingSphereTree::Face face2;
        face2.vertices[0] = vertices[second];
        face2.vertices[1] = vertices[second + 1];
        face2.vertices[2] = vertices[first + 1];
        face2.object_id = 2;
        
        e1 = face2.vertices[1] - face2.vertices[0];
        e2 = face2.vertices[2] - face2.vertices[0];
        face2.normal = e1.cross(e2).normalized();
        sphere_faces_.push_back(face2);
      }
    }
  }
  
  void createSingleTriangle() 
  {
    BoundingSphereTree::Face face;
    face.vertices[0] = Eigen::Vector3f(0, 0, 0);
    face.vertices[1] = Eigen::Vector3f(1, 0, 0);
    face.vertices[2] = Eigen::Vector3f(0, 1, 0);
    face.object_id = 3;
    
    Eigen::Vector3f e1 = face.vertices[1] - face.vertices[0];
    Eigen::Vector3f e2 = face.vertices[2] - face.vertices[0];
    face.normal = e1.cross(e2).normalized();
    
    single_triangle_.push_back(face);
  }
  
  std::vector<BoundingSphereTree::Face> cube_faces_;
  std::vector<BoundingSphereTree::Face> sphere_faces_;
  std::vector<BoundingSphereTree::Face> single_triangle_;
};

TEST_F(BoundingSphereTreeTest, BuildEmptyTree) 
{
  BoundingSphereTree tree;
  std::vector<BoundingSphereTree::Face> empty_faces;
  tree.build(empty_faces);
  
  auto stats = tree.getStats();
  EXPECT_EQ(stats.total_nodes, 0);
  EXPECT_EQ(stats.leaf_nodes, 0);
}

TEST_F(BoundingSphereTreeTest, BuildSingleTriangleTree) 
{
  BoundingSphereTree tree;
  tree.build(single_triangle_);
  
  auto stats = tree.getStats();
  EXPECT_EQ(stats.total_nodes, 1);
  EXPECT_EQ(stats.leaf_nodes, 1);
  EXPECT_EQ(stats.max_depth, 0);
  EXPECT_FLOAT_EQ(stats.average_faces_per_leaf, 1.0f);
}

TEST_F(BoundingSphereTreeTest, BuildCubeTree) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_, 4);
  
  auto stats = tree.getStats();
  EXPECT_GT(stats.total_nodes, 1);
  EXPECT_GT(stats.leaf_nodes, 0);
  EXPECT_LE(stats.average_faces_per_leaf, 4.0f);
}

TEST_F(BoundingSphereTreeTest, BuildLargeSphereTree) 
{
  BoundingSphereTree tree;
  tree.build(sphere_faces_, 8);
  
  auto stats = tree.getStats();
  EXPECT_GT(stats.total_nodes, sphere_faces_.size() / 8);
  EXPECT_GT(stats.max_depth, 2);
}

TEST_F(BoundingSphereTreeTest, SphereIntersectionHit) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);
  
  Sphere test_sphere{Eigen::Vector3f(0, 0, 0), 0.5f};
  EXPECT_TRUE(tree.intersectSphere(test_sphere));
  
  test_sphere = Sphere{Eigen::Vector3f(1.5f, 0, 0), 0.6f};
  EXPECT_TRUE(tree.intersectSphere(test_sphere));
}

TEST_F(BoundingSphereTreeTest, SphereIntersectionMiss) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);
  
  Sphere test_sphere{Eigen::Vector3f(5, 5, 5), 1.0f};
  EXPECT_FALSE(tree.intersectSphere(test_sphere));
  
  test_sphere = Sphere{Eigen::Vector3f(2, 0, 0), 0.9f};
  EXPECT_FALSE(tree.intersectSphere(test_sphere));
}

TEST_F(BoundingSphereTreeTest, CapsuleIntersectionHit) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);
  
  Capsule test_capsule;
  test_capsule.p0 = Eigen::Vector3f(-2, 0, 0);
  test_capsule.p1 = Eigen::Vector3f(2, 0, 0);
  test_capsule.radius = 0.1f;
  EXPECT_TRUE(tree.intersectCapsule(test_capsule));
  
  test_capsule.p0 = Eigen::Vector3f(1, 1, -2);
  test_capsule.p1 = Eigen::Vector3f(1, 1, 2);
  test_capsule.radius = 0.1f;
  EXPECT_TRUE(tree.intersectCapsule(test_capsule));
}

TEST_F(BoundingSphereTreeTest, CapsuleIntersectionWithHitResult) 
{
  BoundingSphereTree tree;
  tree.build(single_triangle_);
  
  Capsule test_capsule;
  test_capsule.p0 = Eigen::Vector3f(0.3f, 0.3f, -1);
  test_capsule.p1 = Eigen::Vector3f(0.3f, 0.3f, 1);
  test_capsule.radius = 0.1f;
  
  collision::HitResult hit_result;
  EXPECT_TRUE(tree.intersectCapsule(test_capsule, hit_result));
  EXPECT_TRUE(hit_result.hit);
  EXPECT_EQ(hit_result.object_id, 3);
  EXPECT_EQ(hit_result.object_type, collision::HitResult::ObjectType::STATIC_OBSTACLE);
  EXPECT_NEAR(hit_result.hit_point.z(), 0.0f, 1e-5f);
}

TEST_F(BoundingSphereTreeTest, CapsuleIntersectionMiss) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);
  
  Capsule test_capsule;
  test_capsule.p0 = Eigen::Vector3f(5, 5, 5);
  test_capsule.p1 = Eigen::Vector3f(6, 6, 6);
  test_capsule.radius = 0.1f;
  EXPECT_FALSE(tree.intersectCapsule(test_capsule));
}

TEST_F(BoundingSphereTreeTest, SegmentIntersection) 
{
  BoundingSphereTree tree;
  tree.build(single_triangle_);
  
  LineSegment seg;
  seg.start = Eigen::Vector3f(0.3f, 0.3f, -1);
  seg.end = Eigen::Vector3f(0.3f, 0.3f, 1);
  EXPECT_TRUE(tree.intersectSegment(seg));
  
  seg.start = Eigen::Vector3f(2, 2, -1);
  seg.end = Eigen::Vector3f(2, 2, 1);
  EXPECT_FALSE(tree.intersectSegment(seg));
}

TEST_F(BoundingSphereTreeTest, DistanceQueries) 
{
  BoundingSphereTree tree;
  tree.build(single_triangle_);
  
  float dist = tree.getDistance(Eigen::Vector3f(0.3f, 0.3f, 0));
  EXPECT_NEAR(dist, 0.0f, 1e-5f);
  
  dist = tree.getDistance(Eigen::Vector3f(0.3f, 0.3f, 1));
  EXPECT_NEAR(dist, 1.0f, 1e-5f);
  
  dist = tree.getDistance(Eigen::Vector3f(0, 0, 0));
  EXPECT_NEAR(dist, 0.0f, 1e-5f);
  
  dist = tree.getDistance(Eigen::Vector3f(2, 0, 0));
  EXPECT_NEAR(dist, 1.0f, 1e-5f);
}

TEST_F(BoundingSphereTreeTest, DistanceToComplexMesh) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);

  float dist_inside = tree.getDistance(Eigen::Vector3f(0, 0, 0));
  EXPECT_NEAR(dist_inside, 0.0f, 1e-5f);

  float dist_outside = tree.getDistance(Eigen::Vector3f(2, 0, 0));
  EXPECT_NEAR(dist_outside, 1.0f, 1e-5f);
  
  float dist_corner = tree.getDistance(Eigen::Vector3f(2, 2, 2));
  float expected_dist_corner = (Eigen::Vector3f(2,2,2) - Eigen::Vector3f(1,1,1)).norm();
  EXPECT_NEAR(dist_corner, expected_dist_corner, 1e-5f);
}

TEST_F(BoundingSphereTreeTest, TreeStatistics) 
{
  BoundingSphereTree tree;
  
  for (uint32_t max_faces : {1, 4, 8, 16}) 
  {
    tree.build(sphere_faces_, max_faces);
    auto stats = tree.getStats();
    
    EXPECT_GT(stats.total_nodes, 0);
    EXPECT_GT(stats.leaf_nodes, 0);
    EXPECT_LE(stats.average_faces_per_leaf, static_cast<float>(max_faces));
    
    size_t expected_max_depth = static_cast<size_t>(
      std::ceil(std::log2(sphere_faces_.size() / max_faces)) + 1
    );
    EXPECT_LE(stats.max_depth, expected_max_depth * 2);
  }
}

TEST_F(BoundingSphereTreeTest, DegenerateCapsule) 
{
  BoundingSphereTree tree;
  tree.build(cube_faces_);
  
  Capsule degen_capsule;
  degen_capsule.p0 = degen_capsule.p1 = Eigen::Vector3f(0, 0, 0);
  degen_capsule.radius = 0.5f;
  EXPECT_TRUE(tree.intersectCapsule(degen_capsule));
  
  degen_capsule.p0 = Eigen::Vector3f(-2, 0, 0);
  degen_capsule.p1 = Eigen::Vector3f(2, 0, 0);
  degen_capsule.radius = 0.0f;
  EXPECT_TRUE(tree.intersectCapsule(degen_capsule));
}

TEST_F(BoundingSphereTreeTest, CoplanarTriangles) 
{
  std::vector<BoundingSphereTree::Face> faces;
  
  for (int i = 0; i < 10; ++i) 
  {
    BoundingSphereTree::Face face;
    float offset = i * 0.5f;
    face.vertices[0] = Eigen::Vector3f(offset, 0, 0);
    face.vertices[1] = Eigen::Vector3f(offset + 1, 0, 0);
    face.vertices[2] = Eigen::Vector3f(offset + 0.5f, 1, 0);
    face.object_id = i;
    face.normal = Eigen::Vector3f(0, 0, 1);
    faces.push_back(face);
  }
  
  BoundingSphereTree tree;
  tree.build(faces);
  
  Sphere sphere{Eigen::Vector3f(2.5f, 0.5f, 0), 0.1f};
  EXPECT_TRUE(tree.intersectSphere(sphere));
}

TEST_F(BoundingSphereTreeTest, PerformanceTest) 
{
  std::vector<BoundingSphereTree::Face> large_mesh;
  const int grid_size = 50;
  
  for (int i = 0; i < grid_size; ++i) 
  {
    for (int j = 0; j < grid_size; ++j) 
    {
      BoundingSphereTree::Face face1, face2;
      
      float x = static_cast<float>(i);
      float y = static_cast<float>(j);
      
      face1.vertices[0] = Eigen::Vector3f(x, y, 0);
      face1.vertices[1] = Eigen::Vector3f(x + 1, y, 0);
      face1.vertices[2] = Eigen::Vector3f(x, y + 1, 0);
      face1.normal = Eigen::Vector3f(0, 0, 1);
      face1.object_id = i * grid_size + j;
      
      face2.vertices[0] = Eigen::Vector3f(x + 1, y, 0);
      face2.vertices[1] = Eigen::Vector3f(x + 1, y + 1, 0);
      face2.vertices[2] = Eigen::Vector3f(x, y + 1, 0);
      face2.normal = Eigen::Vector3f(0, 0, 1);
      face2.object_id = i * grid_size + j;
      
      large_mesh.push_back(face1);
      large_mesh.push_back(face2);
    }
  }
  
  BoundingSphereTree tree;
  auto start = std::chrono::high_resolution_clock::now();
  tree.build(large_mesh, 16);
  auto build_time = std::chrono::high_resolution_clock::now() - start;
  auto build_duration = std::chrono::duration<double, std::milli>(build_time);
  double build_ms = build_duration.count();
  
  EXPECT_LT(build_ms, 1000.0);
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0, grid_size);
  
  const int num_queries = 1000;
  start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < num_queries; ++i) 
  {
    Sphere test_sphere
    {
      Eigen::Vector3f(dist(gen), dist(gen), 0),
      0.5f
    };
    tree.intersectSphere(test_sphere);
  }
  
  auto query_time = std::chrono::high_resolution_clock::now() - start;
  auto avg_duration = std::chrono::duration<double, std::milli>(query_time);
  double avg_query_ms = avg_duration.count() / num_queries;
  
  EXPECT_LT(avg_query_ms, 1.0);
}

TEST(BSTBuilderTest, BuildFromManualFaces) 
{
  GeometryConfig config;
  config.max_faces_per_leaf = 8;
  
  BSTBuilder builder(config);
  
  builder.addFace(
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(1, 0, 0),
    Eigen::Vector3f(0, 1, 0),
    1
  );
  
  builder.addFace(
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(1, 0, 0),
    Eigen::Vector3f(0, 0, 1),
    1
  );
  
  builder.addFace(
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0, 1, 0),
    Eigen::Vector3f(0, 0, 1),
    1
  );
  
  builder.addFace(
    Eigen::Vector3f(1, 0, 0),
    Eigen::Vector3f(0, 1, 0),
    Eigen::Vector3f(0, 0, 1),
    1
  );
  
  EXPECT_EQ(builder.getFaceCount(), 4);
  
  auto tree = builder.build();
  ASSERT_NE(tree, nullptr);
  
  auto stats = tree->getStats();
  EXPECT_EQ(stats.total_nodes, 1);
  EXPECT_EQ(stats.leaf_nodes, 1);
}

TEST(BSTBuilderTest, LoadOBJFile) 
{
  GeometryConfig config;
  BSTBuilder builder(config);
  
  std::string obj_content = R"(
v -1 -1 -1
v 1 -1 -1
v 1 1 -1
v -1 1 -1
v -1 -1 1
v 1 -1 1
v 1 1 1
v -1 1 1

f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 4 8
f 1 8 5
f 2 6 7
f 2 7 3
f 1 5 6
f 1 6 2
f 4 3 7
f 4 7 8
)";
  
  std::string temp_file = "test_cube.obj";
  std::ofstream file(temp_file);
  file << obj_content;
  file.close();
  
  EXPECT_TRUE(builder.loadMesh(temp_file, 1));
  EXPECT_EQ(builder.getFaceCount(), 12);
  
  auto tree = builder.build();
  ASSERT_NE(tree, nullptr);
  
  std::remove(temp_file.c_str());
}

TEST(BSTBuilderTest, ClearFaces) 
{
  GeometryConfig config;
  BSTBuilder builder(config);
  
  builder.addFace(
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(1, 0, 0),
    Eigen::Vector3f(0, 1, 0),
    1
  );
  
  EXPECT_EQ(builder.getFaceCount(), 1);
  
  builder.clear();
  EXPECT_EQ(builder.getFaceCount(), 0);
  
  auto tree = builder.build();
  EXPECT_EQ(tree, nullptr);
}

}
}
}