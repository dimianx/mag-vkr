#include <gtest/gtest.h>
#include "vkr/geometry/collision/uniform_grid.hpp"
#include <random>
#include <thread>
#include <atomic>

namespace vkr
{

namespace geometry
{

namespace collision
{
class UniformGridTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    cell_size_ = 1.0f;
    bounds_ = BoundingBox
    {
      Eigen::Vector3f(-10, -10, -10),
      Eigen::Vector3f(10, 10, 10)
    };
    hash_table_size_ = 1024;
  }
  
  float cell_size_;
  BoundingBox bounds_;
  uint32_t hash_table_size_;
};

TEST_F(UniformGridTest, Construction) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 0);
  EXPECT_EQ(stats.occupied_cells, 0);
  EXPECT_EQ(stats.average_objects_per_cell, 0.0f);
}

TEST_F(UniformGridTest, ConstructionNonPowerOf2) 
{
  UniformGrid grid(cell_size_, bounds_, 1000);
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 0);
}

TEST_F(UniformGridTest, InsertSingleObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox obj_bbox
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  
  grid.insert(1, obj_bbox);
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 1);
  EXPECT_EQ(stats.occupied_cells, 1);
  EXPECT_EQ(stats.average_objects_per_cell, 1.0f);
}

TEST_F(UniformGridTest, InsertObjectSpanningMultipleCells) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox obj_bbox
  {
    Eigen::Vector3f(-0.5f, -0.5f, -0.5f),
    Eigen::Vector3f(1.5f, 1.5f, 1.5f)
  };
  
  grid.insert(1, obj_bbox);
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 1);
  EXPECT_GT(stats.occupied_cells, 1);
  EXPECT_GT(stats.average_objects_per_cell, 0.0f);
}

TEST_F(UniformGridTest, InsertMultipleObjects) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  for (int i = 0; i < 10; ++i) 
  {
    float pos = static_cast<float>(i * 2);
    BoundingBox bbox
    {
      Eigen::Vector3f(pos, 0, 0),
      Eigen::Vector3f(pos + 0.5f, 0.5f, 0.5f)
    };
    grid.insert(i, bbox);
  }
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 10);
  
  for (int i = 0; i < 10; ++i)
  {
    float pos = static_cast<float>(i * 2);
    BoundingBox query_bbox
    {
      Eigen::Vector3f(pos, 0, 0),
      Eigen::Vector3f(pos + 0.5f, 0.5f, 0.5f)
    };
    auto results = grid.query(query_bbox);
    
    bool found = false;
    for (ObjectId id : results) 
    {
      if (id == static_cast<ObjectId>(i)) 
      {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Object with id " << i << " was not found.";
  }
}

TEST_F(UniformGridTest, RemoveObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox bbox
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  
  grid.insert(1, bbox);
  EXPECT_EQ(grid.getStats().total_objects, 1);
  
  grid.remove(1);
  EXPECT_EQ(grid.getStats().total_objects, 0);
  EXPECT_EQ(grid.getStats().occupied_cells, 0);
}

TEST_F(UniformGridTest, RemoveNonExistentObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  grid.remove(999);
  
  EXPECT_EQ(grid.getStats().total_objects, 0);
}

TEST_F(UniformGridTest, QueryEmptyGrid) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox query_bbox
  {
    Eigen::Vector3f(-1, -1, -1),
    Eigen::Vector3f(1, 1, 1)
  };
  
  auto results = grid.query(query_bbox);
  EXPECT_TRUE(results.empty());
}

TEST_F(UniformGridTest, QuerySingleObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox obj_bbox
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  grid.insert(1, obj_bbox);
  
  BoundingBox query_bbox
  {
    Eigen::Vector3f(0.25f, 0.25f, 0.25f),
    Eigen::Vector3f(1, 1, 1)
  };
  
  auto results = grid.query(query_bbox);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], 1);
  
  query_bbox = BoundingBox
  {
    Eigen::Vector3f(2, 2, 2),
    Eigen::Vector3f(3, 3, 3)
  };
  
  results = grid.query(query_bbox);
  EXPECT_TRUE(results.empty());
}

TEST_F(UniformGridTest, QueryMultipleObjects) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  for (int i = 0; i < 3; ++i) 
  {
    for (int j = 0; j < 3; ++j) 
    {
      BoundingBox bbox
      {
        Eigen::Vector3f(i * 2.0f, j * 2.0f, 0),
        Eigen::Vector3f(i * 2.0f + 0.5f, j * 2.0f + 0.5f, 0.5f)
      };
      grid.insert(i * 3 + j, bbox);
    }
  }
  
  BoundingBox query_bbox
  {
    Eigen::Vector3f(-1, -1, -1),
    Eigen::Vector3f(5, 5, 1)
  };
  
  auto results = grid.query(query_bbox);
  EXPECT_EQ(results.size(), 9);
}

TEST_F(UniformGridTest, QueryRayHit) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox obj_bbox
  {
    Eigen::Vector3f(-0.5f, -0.5f, -0.5f),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  grid.insert(1, obj_bbox);
  
  Eigen::Vector3f origin(-2, 0, 0);
  Eigen::Vector3f direction(1, 0, 0);
  
  auto results = grid.queryRay(origin, direction, 5.0f);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], 1);
}

TEST_F(UniformGridTest, QueryRayMiss) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox obj_bbox
  {
    Eigen::Vector3f(-0.5f, -0.5f, -0.5f),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  grid.insert(1, obj_bbox);
  
  Eigen::Vector3f origin(-2, 2, 0);
  Eigen::Vector3f direction(1, 0, 0);
  
  auto results = grid.queryRay(origin, direction, 5.0f);
  EXPECT_TRUE(results.empty());
}

TEST_F(UniformGridTest, QueryRayFirstHit) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  for (int i = 0; i < 5; ++i) 
  {
    float x = static_cast<float>(i * 2);
    BoundingBox bbox
    {
      Eigen::Vector3f(x - 0.5f, -0.5f, -0.5f),
      Eigen::Vector3f(x + 0.5f, 0.5f, 0.5f)
    };
    grid.insert(i, bbox);
  }
  
  Eigen::Vector3f origin(-2, 0, 0);
  Eigen::Vector3f direction(1, 0, 0);
  
  ObjectId first_hit = grid.queryRayFirstHit(origin, direction, 20.0f);
  EXPECT_EQ(first_hit, 0);
}

TEST_F(UniformGridTest, UpdateObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox initial_bbox
  {
    Eigen::Vector3f(0, 0, 0),
    Eigen::Vector3f(0.5f, 0.5f, 0.5f)
  };
  
  grid.insert(1, initial_bbox);
  
  BoundingBox new_bbox
  {
    Eigen::Vector3f(5, 5, 5),
    Eigen::Vector3f(5.5f, 5.5f, 5.5f)
  };
  
  grid.update(1, new_bbox);
  
  auto results = grid.query(initial_bbox);
  EXPECT_TRUE(results.empty());
  
  results = grid.query(new_bbox);
  EXPECT_EQ(results.size(), 1);
  EXPECT_EQ(results[0], 1);
}

TEST_F(UniformGridTest, Clear) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  for (int i = 0; i < 10; ++i) 
  {
    BoundingBox bbox
    {
      Eigen::Vector3f(i, 0, 0),
      Eigen::Vector3f(i + 0.5f, 0.5f, 0.5f)
    };
    grid.insert(i, bbox);
  }
  
  EXPECT_EQ(grid.getStats().total_objects, 10);
  
  grid.clear();
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 0);
  EXPECT_EQ(stats.occupied_cells, 0);
}

TEST_F(UniformGridTest, ConcurrentInsertions) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  const int num_threads = 4;
  const int objects_per_thread = 100;
  
  std::vector<std::thread> threads;
  
  for (int t = 0; t < num_threads; ++t) 
  {
    threads.emplace_back([&grid, t, objects_per_thread]() 
    {
      for (int i = 0; i < objects_per_thread; ++i) 
      {
        ObjectId id = t * objects_per_thread + i;
        float pos = static_cast<float>(id);
        
        BoundingBox bbox
        {
          Eigen::Vector3f(pos * 0.1f, 0, 0),
          Eigen::Vector3f(pos * 0.1f + 0.05f, 0.05f, 0.05f)
        };
        
        grid.insert(id, bbox);
      }
    });
  }
  
  for (auto& thread : threads) 
  {
    thread.join();
  }
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, num_threads * objects_per_thread);
}

TEST_F(UniformGridTest, ConcurrentQueries) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  for (int i = 0; i < 100; ++i) 
  {
    float pos = static_cast<float>(i % 10);
    BoundingBox bbox
    {
      Eigen::Vector3f(pos, pos, 0),
      Eigen::Vector3f(pos + 0.5f, pos + 0.5f, 0.5f)
    };
    grid.insert(i, bbox);
  }
  
  std::atomic<int> total_results(0);
  const int num_threads = 4;
  std::vector<std::thread> threads;
  
  for (int t = 0; t < num_threads; ++t) 
  {
    threads.emplace_back([&grid, &total_results]() 
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-5, 15);
      
      for (int i = 0; i < 100; ++i) 
      {
        BoundingBox query_bbox
        {
          Eigen::Vector3f(dist(gen), dist(gen), -1),
          Eigen::Vector3f(dist(gen) + 2, dist(gen) + 2, 1)
        };
        
        auto results = grid.query(query_bbox);
        total_results += static_cast<int>(results.size());
      }
    });
  }
  
  for (auto& thread : threads) 
  {
    thread.join();
  }
  
  EXPECT_GT(total_results.load(), 0);
}

TEST_F(UniformGridTest, ObjectAtGridBoundary) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox bbox
  {
    Eigen::Vector3f(9.5f, 9.5f, 9.5f),
    Eigen::Vector3f(10.0f, 10.0f, 10.0f)
  };
  
  grid.insert(1, bbox);
  
  auto results = grid.query(bbox);
  EXPECT_EQ(results.size(), 1);
}

TEST_F(UniformGridTest, VeryLargeObject) 
{
  UniformGrid grid(cell_size_, bounds_, hash_table_size_);
  
  BoundingBox bbox
  {
    Eigen::Vector3f(-9, -9, -9),
    Eigen::Vector3f(9, 9, 9)
  };
  
  grid.insert(1, bbox);
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 1);
  EXPECT_GT(stats.occupied_cells, 100);
}

TEST_F(UniformGridTest, HashCollisions) 
{
  UniformGrid grid(cell_size_, bounds_, 16);
  
  for (int i = 0; i < 100; ++i) 
  {
    float pos = static_cast<float>(i);
    BoundingBox bbox
    {
      Eigen::Vector3f(pos, 0, 0),
      Eigen::Vector3f(pos + 0.1f, 0.1f, 0.1f)
    };
    grid.insert(i, bbox);
  }
  
  auto stats = grid.getStats();
  EXPECT_EQ(stats.total_objects, 100);
  EXPECT_GT(stats.max_objects_in_cell, 1);
  
  for (int i = 0; i < 100; ++i) 
  {
    float pos = static_cast<float>(i);
    BoundingBox query
    {
      Eigen::Vector3f(pos, 0, 0),
      Eigen::Vector3f(pos + 0.1f, 0.1f, 0.1f)
    };
    
    auto results = grid.query(query);
    EXPECT_FALSE(results.empty());
    
    bool found = false;
    for (ObjectId id : results) 
    {
      if (id == static_cast<ObjectId>(i)) 
      {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST_F(UniformGridTest, PerformanceTest) 
{
  UniformGrid grid(0.5f, bounds_, 4096);
  
  const int num_objects = 10000;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> pos_dist(-8, 8);
  std::uniform_real_distribution<float> size_dist(0.1f, 0.5f);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < num_objects; ++i) 
  {
    Eigen::Vector3f pos(pos_dist(gen), pos_dist(gen), pos_dist(gen));
    float size = size_dist(gen);
    
    BoundingBox bbox
    {
      pos,
      pos + Eigen::Vector3f(size, size, size)
    };
    
    grid.insert(i, bbox);
  }
  
  auto insert_time = std::chrono::high_resolution_clock::now() - start;
  double insert_ms = std::chrono::duration<double, std::milli>(insert_time).count();
  
  EXPECT_LT(insert_ms / num_objects, 0.1);
  
  const int num_queries = 1000;
  int total_results = 0;
  
  start = std::chrono::high_resolution_clock::now();
  
  for (int i = 0; i < num_queries; ++i) 
  {
    Eigen::Vector3f center(pos_dist(gen), pos_dist(gen), pos_dist(gen));
    
    BoundingBox query_bbox
    {
      center - Eigen::Vector3f(1, 1, 1),
      center + Eigen::Vector3f(1, 1, 1)
    };
    
    auto results = grid.query(query_bbox);
    total_results += static_cast<int>(results.size());
  }
  
  auto query_time = std::chrono::high_resolution_clock::now() - start;
  double query_ms = std::chrono::duration<double, std::milli>(query_time).count();
  
  EXPECT_LT(query_ms / num_queries, 1.0);
  
  EXPECT_GT(total_results, 0);
}

}
}
}