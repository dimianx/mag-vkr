#include <gtest/gtest.h>
#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/terrain/qwg_builder.hpp"
#include "vkr/geometry/primitives.hpp"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <random>
#include <gdal_priv.h>
#include <thread>
#include <chrono>
#include <functional>

namespace vkr 
{
namespace terrain 
{

namespace fs = std::filesystem;

class WaveletGridTest : public ::testing::Test 
{
protected:
  static void SetUpTestSuite() 
  {
    rng_.seed(42);
    
    test_dir_ = fs::current_path() / "tests" / "data" / "temp_wavelet_grid";
    fs::create_directories(test_dir_);
    
    GDALAllRegister();
    
    createTestFiles();
  }
  
  static void TearDownTestSuite() 
  {
    if (fs::exists(test_dir_)) 
    {
      fs::remove_all(test_dir_);
    }
  }
  
  static void createTestFiles() 
  {
    createMainTestQWG();
    
    createNodataTestQWG();
  }
  
  static void createMainTestQWG() 
  {
    constexpr size_t size = 256;
    std::string tif_file = (test_dir_ / "test_terrain.tif").string();
    
    createTestGeoTIFF(tif_file, size, size, [](size_t x, size_t y) {
      float base = 100.0f;
      float hill1 = 50.0f * std::exp(-((x-64.f)*(x-64.f) + (y-64.f)*(y-64.f)) / 1000.0f);
      float hill2 = 30.0f * std::exp(-((x-192.f)*(x-192.f) + (y-192.f)*(y-192.f)) / 800.0f);
      float noise = noise_dist_(rng_);
      return base + hill1 + hill2 + noise;
    });
    
    test_qwg_file_ = (test_dir_ / "test_terrain.qwg").string();
    
    TerrainConfig config;
    config.wavelet_levels = 4;
    config.page_size = 4096;
    config.max_cache_size = 10;
    
    QWGBuilder builder(config);
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(tif_file, test_qwg_file_, error_message);
    if (!success) 
    {
      throw std::runtime_error("Failed to create test QWG: " + error_message);
    }
  }
  
  static void createNodataTestQWG() 
  {
    std::string tif_file = (test_dir_ / "nodata_terrain.tif").string();
    
    createTestGeoTIFF(tif_file, 128, 128, [](size_t x, size_t y) {
      if ((x / 32 + y / 32) % 2 == 0) 
      {
        return -9999.0f;
      }
      return 100.0f + x * 0.1f + y * 0.1f;
    });
    
    nodata_qwg_file_ = (test_dir_ / "nodata_terrain.qwg").string();
    
    TerrainConfig config;
    QWGBuilder builder(config);
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(tif_file, nodata_qwg_file_, error_message);
    if (!success) 
    {
      throw std::runtime_error("Failed to create nodata QWG: " + error_message);
    }
  }
  
  static void createTestGeoTIFF(const std::string& filename, size_t width, size_t height,
                                std::function<float(size_t, size_t)> generator) 
  {
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver)
    {
      throw std::runtime_error("GTiff driver not available");
    }
    
    GDALDataset* dataset = driver->Create(filename.c_str(), width, height, 
                                         1, GDT_Float32, nullptr);
    if (!dataset)
    {
      throw std::runtime_error("Failed to create GeoTIFF dataset");
    }
    
    double geotransform[6] = {1000, 1, 0, 2000, 0, -1};
    dataset->SetGeoTransform(geotransform);
    
    std::vector<float> buffer(width * height);
    for (size_t y = 0; y < height; ++y) 
    {
      for (size_t x = 0; x < width; ++x) 
      {
        buffer[y * width + x] = generator(x, y);
      }
    }
    
    GDALRasterBand* band = dataset->GetRasterBand(1);
    CPLErr err = band->RasterIO(GF_Write, 0, 0, width, height, 
                               buffer.data(), width, height, GDT_Float32, 0, 0);
    if (err != CE_None)
    {
      GDALClose(dataset);
      throw std::runtime_error("Failed to write raster data");
    }
    band->SetNoDataValue(-9999.0);
    
    GDALFlushCache(dataset);
    GDALClose(dataset);
  }
  
  static std::mt19937 rng_;
  static std::uniform_real_distribution<float> noise_dist_;
  static fs::path test_dir_;
  static std::string test_qwg_file_;
  static std::string nodata_qwg_file_;
};

std::mt19937 WaveletGridTest::rng_;
std::uniform_real_distribution<float> WaveletGridTest::noise_dist_{-5.0f, 5.0f};
fs::path WaveletGridTest::test_dir_;
std::string WaveletGridTest::test_qwg_file_;
std::string WaveletGridTest::nodata_qwg_file_;

TEST_F(WaveletGridTest, LoadFromFile) 
{
  TerrainConfig config;
  config.max_cache_size = 10;
  
  WaveletGrid grid(config);
  
  EXPECT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  EXPECT_FALSE(grid.loadFromFile("non_existent.qwg"));
}

TEST_F(WaveletGridTest, HeightQueries) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  struct TestPoint 
  {
    float x, y;
    float min_expected, max_expected;
  };
  
  std::vector<TestPoint> test_points = {
    {1064.0f, 1936.0f, 140.0f, 160.0f},
    {1192.0f, 1808.0f, 120.0f, 140.0f},
    {1128.0f, 1872.0f, 95.0f, 110.0f},
    {1000.0f, 2000.0f, 95.0f, 105.0f},
  };
  
  for (const auto& pt : test_points) 
  {
    float height = grid.getHeight(pt.x, pt.y);
    EXPECT_GE(height, pt.min_expected) << "Point (" << pt.x << ", " << pt.y << ")";
    EXPECT_LE(height, pt.max_expected) << "Point (" << pt.x << ", " << pt.y << ")";
  }
}

TEST_F(WaveletGridTest, OutOfBoundsQueries) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  EXPECT_EQ(grid.getHeight(500.0f, 1500.0f), 0.0f);
  EXPECT_EQ(grid.getHeight(1500.0f, 2500.0f), 0.0f);
  EXPECT_EQ(grid.getHeight(2000.0f, 1500.0f), 0.0f);
  EXPECT_EQ(grid.getHeight(1500.0f, 1000.0f), 0.0f);
}

TEST_F(WaveletGridTest, Interpolation) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  float h1 = grid.getHeight(1050.0f, 1950.0f);
  float h2 = grid.getHeight(1050.5f, 1950.0f);
  float h3 = grid.getHeight(1051.0f, 1950.0f);
  
  if (h1 < h3) 
  {
    EXPECT_GE(h2, h1);
    EXPECT_LE(h2, h3);
  } 
  else 
  {
    EXPECT_LE(h2, h1);
    EXPECT_GE(h2, h3);
  }
  
  EXPECT_NEAR(h2, (h1 + h3) / 2.0f, 5.0f);
}

TEST_F(WaveletGridTest, MinMaxQueries) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  geometry::BoundingBox small_box;
  small_box.min = Eigen::Vector3f(1060, 1930, 0);
  small_box.max = Eigen::Vector3f(1070, 1940, 200);
  
  MinMax small_range = grid.getMinMax(small_box);
  EXPECT_LT(small_range.min, small_range.max);
  EXPECT_GE(small_range.min, 95.0f); 
  EXPECT_LE(small_range.max, 200.0f);
  
  geometry::BoundingBox large_box;
  large_box.min = Eigen::Vector3f(1000, 1744, 0);
  large_box.max = Eigen::Vector3f(1256, 2000, 300);
  
  MinMax large_range = grid.getMinMax(large_box);
  EXPECT_LT(large_range.min, large_range.max);
  EXPECT_LE(large_range.min, small_range.min);
  EXPECT_GE(large_range.max, small_range.max);
}

TEST_F(WaveletGridTest, BatchQueries) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  constexpr size_t count = 1000;
  std::vector<float> x_coords(count);
  std::vector<float> y_coords(count);
  std::vector<float> heights_batch(count);
  std::vector<float> heights_single(count);
  
  std::uniform_real_distribution<float> coord_dist(1000.0f, 1256.0f);
  
  for (size_t i = 0; i < count; ++i) 
  {
    x_coords[i] = coord_dist(rng_);
    y_coords[i] = 2000.0f - coord_dist(rng_) + 1000.0f;
  }
  
  grid.batchGetHeight(x_coords.data(), y_coords.data(), count, heights_batch.data());
  
  for (size_t i = 0; i < count; ++i) 
  {
    heights_single[i] = grid.getHeight(x_coords[i], y_coords[i]);
  }
  
  for (size_t i = 0; i < count; ++i) 
  {
    EXPECT_NEAR(heights_batch[i], heights_single[i], 1e-5f) 
        << "Mismatch at index " << i;
  }
}

TEST_F(WaveletGridTest, CacheBehavior) 
{
  TerrainConfig config;
  config.max_cache_size = 1;
  
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  std::vector<float> x_coords = {1010, 1200, 1010, 1200, 1010};
  std::vector<float> y_coords = {1990, 1810, 1990, 1810, 1990};
  
  for (size_t i = 0; i < x_coords.size(); ++i) 
  {
    float height = grid.getHeight(x_coords[i], y_coords[i]);
    EXPECT_GT(height, 0.0f);
  }
  
  float h1_first = grid.getHeight(1010, 1990);
  float h1_last = grid.getHeight(1010, 1990);
  EXPECT_EQ(h1_first, h1_last);
}

TEST_F(WaveletGridTest, GetBounds) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  geometry::BoundingBox bounds = grid.getBounds();
  
  EXPECT_NEAR(bounds.min.x(), 1000.0f, 1.0f);
  EXPECT_NEAR(bounds.min.y(), 1744.0f, 1.0f);
  EXPECT_GT(bounds.min.z(), 90.0f);
  
  EXPECT_NEAR(bounds.max.x(), 1256.0f, 1.0f);
  EXPECT_NEAR(bounds.max.y(), 2000.0f, 1.0f);
  EXPECT_LT(bounds.max.z(), 200.0f);
}

TEST_F(WaveletGridTest, JSONExport) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  std::string json_file = (test_dir_ / "terrain_export.json").string();
  
  grid.exportToJSON(json_file);
  
  EXPECT_TRUE(fs::exists(json_file));
  EXPECT_GT(fs::file_size(json_file), 100u);
}

TEST_F(WaveletGridTest, HeightConsistency) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(test_qwg_file_));
  
  float x = 1128.0f;
  float y = 1872.0f;
  
  std::vector<float> heights;
  for (int i = 0; i < 10; ++i) 
  {
    heights.push_back(grid.getHeight(x, y));
  }
  
  for (size_t i = 1; i < heights.size(); ++i) 
  {
    EXPECT_EQ(heights[0], heights[i]) << "Query " << i << " differs";
  }
  
  for (int tx = 0; tx < 10; ++tx) 
  {
    for (int ty = 0; ty < 10; ++ty) 
    {
      grid.getHeight(1000.0f + tx * 25.0f, 2000.0f - ty * 25.0f);
    }
  }
  
  float height_after = grid.getHeight(x, y);
  EXPECT_EQ(heights[0], height_after);
}

TEST_F(WaveletGridTest, NodataHandling) 
{
  TerrainConfig config;
  WaveletGrid grid(config);
  ASSERT_TRUE(grid.loadFromFile(nodata_qwg_file_));
  
  float height_nodata = grid.getHeight(1016.0f, 1984.0f);
  EXPECT_NE(height_nodata, -9999.0f);
  EXPECT_GT(height_nodata, 50.0f);
  
  float height_valid = grid.getHeight(1048.0f, 1952.0f);
  EXPECT_GT(height_valid, 100.0f);
}

TEST_F(WaveletGridTest, MoveSemantics) 
{
  TerrainConfig config;
  WaveletGrid grid1(config);
  ASSERT_TRUE(grid1.loadFromFile(test_qwg_file_));
  
  float h1 = grid1.getHeight(1100.0f, 1900.0f);
  
  WaveletGrid grid2(std::move(grid1));
  
  float h2 = grid2.getHeight(1100.0f, 1900.0f);
  EXPECT_EQ(h1, h2);
  
  WaveletGrid grid3(config);
  grid3 = std::move(grid2);
  
  float h3 = grid3.getHeight(1100.0f, 1900.0f);
  EXPECT_EQ(h1, h3);
}

}
}