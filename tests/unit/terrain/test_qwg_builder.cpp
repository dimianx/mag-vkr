#include <gtest/gtest.h>
#include "vkr/terrain/qwg_builder.hpp"
#include <fstream>
#include <filesystem>
#include <random>
#include <gdal_priv.h>
#include <cpl_conv.h>

namespace vkr 
{
namespace terrain 
{


namespace fs = std::filesystem;

class QWGBuilderTest : public ::testing::Test 
{
protected:
  void SetUp() override 
  {
    rng_.seed(42);
    
    test_dir_ = fs::current_path() / "tests" / "data" / "temp";
    fs::create_directories(test_dir_);
    
    GDALAllRegister();
  }
  
  void TearDown() override 
  {
    for (const auto& file : temp_files_) 
    {
      if (fs::exists(file)) 
      {
        fs::remove(file);
      }
    }
  }
  
  std::string createTestGeoTIFF(size_t width, size_t height, 
                               std::function<float(size_t, size_t)> generator,
                               const std::string& name = "test.tif") 
  {
    std::string filename = (test_dir_ / name).string();
    temp_files_.push_back(filename);
    
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) 
    {
      throw std::runtime_error("GTiff driver not available");
    }
    
    GDALDataset* dataset = driver->Create(filename.c_str(), width, height, 
                                         1, GDT_Float32, nullptr);
    if (!dataset) 
    {
      throw std::runtime_error("Failed to create GeoTIFF");
    }
    
    double geotransform[6] = {0, 1, 0, 0, 0, -1};
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
    band->RasterIO(GF_Write, 0, 0, width, height, 
                   buffer.data(), width, height, GDT_Float32, 0, 0);
    
    band->SetNoDataValue(-9999.0);
    
    GDALClose(dataset);
    return filename;
  }
  
  #pragma pack(push, 1)
  struct QWGHeader 
  {
    uint32_t magic;
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t tile_size;
    uint32_t wavelet_levels;
    float origin_x;
    float origin_y;
    float cell_size;
    float global_min_height;
    float global_max_height;
    uint32_t num_tiles;
    uint32_t tile_index_offset;
    uint32_t tile_data_offset;
    uint32_t wavelet_type;
    float nodata_value;
    uint32_t reserved[8];
  };
  #pragma pack(pop)
  
  bool readQWGHeader(const std::string& filename, QWGHeader& header) 
  {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(&header), sizeof(QWGHeader));
    return file.good();
  }
  
  std::mt19937 rng_;
  std::uniform_real_distribution<float> height_dist_{0.0f, 1000.0f};
  fs::path test_dir_;
  std::vector<std::string> temp_files_;
};

TEST_F(QWGBuilderTest, BasicConversion) 
{
  constexpr size_t size = 256;
  auto input_file = createTestGeoTIFF(size, size, [this](size_t, size_t) {
    return height_dist_(rng_);
  });
  
  std::string output_file = (test_dir_ / "output.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  config.wavelet_levels = 4;
  
  QWGBuilder builder(config);
  std::string error_message;
  
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
  EXPECT_TRUE(success) << "Error: " << error_message;
  
  EXPECT_TRUE(fs::exists(output_file));
  
  QWGHeader header;
  EXPECT_TRUE(readQWGHeader(output_file, header));
  EXPECT_EQ(header.magic, 0x47574851);
  EXPECT_EQ(header.version, 1);
  EXPECT_EQ(header.wavelet_levels, 4);
}

TEST_F(QWGBuilderTest, DifferentWavelets) 
{
  constexpr size_t size = 128;
  auto input_file = createTestGeoTIFF(size, size, [](size_t x, size_t y) {
    return 100.0f + x * 0.5f + y * 0.5f;
  });
  
  TerrainConfig config;
  config.wavelet_levels = 3;
  
  std::vector<QWGBuilder::WaveletType> types = {
    QWGBuilder::WaveletType::HAAR,
  };
  
  std::vector<size_t> file_sizes;
  
  for (auto type : types) 
  {
    QWGBuilder builder(config);
    builder.setWaveletType(type);
    
    std::string output_file = (test_dir_ / ("wavelet_" + std::to_string(static_cast<int>(type)) + ".qwg")).string();
    temp_files_.push_back(output_file);
    
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
    EXPECT_TRUE(success) << "Type: " << static_cast<int>(type) << ", Error: " << error_message;
    
    file_sizes.push_back(fs::file_size(output_file));
  }
  
  for (size_t size : file_sizes) 
  {
    EXPECT_GT(size, 0u);
  }
}

TEST_F(QWGBuilderTest, ProgressCallback) 
{
  constexpr size_t size = 512;
  auto input_file = createTestGeoTIFF(size, size, [this](size_t, size_t) {
    return height_dist_(rng_);
  });
  
  std::string output_file = (test_dir_ / "progress.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  QWGBuilder builder(config);
  
  std::vector<size_t> progress_values;
  std::vector<std::string> progress_messages;
  bool completed = false;
  
  auto callback = [&](size_t current, size_t total, const std::string& msg) {
    progress_values.push_back(current);
    progress_messages.push_back(msg);
    if (current == total) 
    {
      completed = true;
    }
  };
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message, callback);
  EXPECT_TRUE(success);
  
  EXPECT_FALSE(progress_values.empty());
  EXPECT_FALSE(progress_messages.empty());
  EXPECT_TRUE(completed);
  
  for (size_t i = 1; i < progress_values.size(); ++i) 
  {
    EXPECT_GE(progress_values[i], progress_values[i-1]);
  }
}

TEST_F(QWGBuilderTest, NodataHandling) 
{
  constexpr size_t size = 128;
  constexpr float nodata = -9999.0f;
  
  auto input_file = createTestGeoTIFF(size, size, [nodata](size_t x, size_t y) {
    if ((x + y) % 10 == 0) 
    {
      return nodata;
    }
    return 100.0f + x * 0.1f + y * 0.1f;
  });
  
  std::string output_file = (test_dir_ / "nodata.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  QWGBuilder builder(config);
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
  EXPECT_TRUE(success);
  
  QWGHeader header;
  EXPECT_TRUE(readQWGHeader(output_file, header));
  
  EXPECT_NEAR(header.nodata_value, nodata, 1e-6f);
}

TEST_F(QWGBuilderTest, LargeFileHandling) 
{
  constexpr size_t size = 1024;
  auto input_file = createTestGeoTIFF(size, size, [this](size_t, size_t) {
    return height_dist_(rng_);
  }, "large.tif");
  
  std::string output_file = (test_dir_ / "large.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  config.wavelet_levels = 5;
  config.page_size = 4096;
  
  QWGBuilder builder(config);
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
  EXPECT_TRUE(success);
  
  QWGHeader header;
  EXPECT_TRUE(readQWGHeader(output_file, header));
  EXPECT_GT(header.num_tiles, 1u);
  
  EXPECT_GE(header.tile_size, 64u);
  EXPECT_LE(header.tile_size, 2048u);
}

TEST_F(QWGBuilderTest, InvalidInputHandling) 
{
  TerrainConfig config;
  QWGBuilder builder(config);
  std::string error_message;
  
  {
    bool success = builder.buildFromGeoTIFF("non_existent.tif", "output.qwg", error_message);
    EXPECT_FALSE(success);
    EXPECT_FALSE(error_message.empty());
  }
  
  {
    auto input_file = createTestGeoTIFF(64, 64, [](size_t, size_t) { return 100.0f; });
    bool success = builder.buildFromGeoTIFF(input_file, "/invalid/path/output.qwg", error_message);
    EXPECT_FALSE(success);
  }
}

TEST_F(QWGBuilderTest, TileBoundaries) 
{
  constexpr size_t width = 250;
  constexpr size_t height = 130;
  
  auto input_file = createTestGeoTIFF(width, height, [](size_t x, size_t y) {
    return (x % 64) * (y % 64);
  });
  
  std::string output_file = (test_dir_ / "boundaries.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  config.wavelet_levels = 4;
  
  QWGBuilder builder(config);
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
  EXPECT_TRUE(success);
  
  QWGHeader header;
  EXPECT_TRUE(readQWGHeader(output_file, header));
  
  EXPECT_GE(header.width, width);
  EXPECT_GE(header.height, height);
  EXPECT_EQ(header.width % header.tile_size, 0u);
  EXPECT_EQ(header.height % header.tile_size, 0u);
}

TEST_F(QWGBuilderTest, GlobalMinMaxComputation) 
{
  constexpr size_t size = 256;
  constexpr float min_height = 100.0f;
  constexpr float max_height = 500.0f;
  constexpr float nodata = -9999.0f;
  
  auto input_file = createTestGeoTIFF(size, size, [=](size_t x, size_t y) {
    if (x < 10 && y < 10) return nodata;
    
    float t = (x + y) / float(2 * size);
    return min_height + t * (max_height - min_height);
  });
  
  std::string output_file = (test_dir_ / "minmax.qwg").string();
  temp_files_.push_back(output_file);
  
  TerrainConfig config;
  QWGBuilder builder(config);
  
  std::string error_message;
  bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
  EXPECT_TRUE(success);
  
  QWGHeader header;
  EXPECT_TRUE(readQWGHeader(output_file, header));
  
  EXPECT_NEAR(header.global_min_height, min_height, 10.0f);
  EXPECT_NEAR(header.global_max_height, max_height, 10.0f);
}

TEST_F(QWGBuilderTest, WaveletLevels) 
{
  constexpr size_t size = 256;
  auto input_file = createTestGeoTIFF(size, size, [this](size_t, size_t) {
    return height_dist_(rng_);
  });
  
  std::vector<uint8_t> levels = {1, 2, 4, 6, 8};
  
  for (uint8_t level : levels) 
  {
    TerrainConfig config;
    config.wavelet_levels = level;
    
    QWGBuilder builder(config);
    
    std::string output_file = (test_dir_ / ("levels_" + std::to_string(level) + ".qwg")).string();
    temp_files_.push_back(output_file);
    
    std::string error_message;
    bool success = builder.buildFromGeoTIFF(input_file, output_file, error_message);
    EXPECT_TRUE(success) << "Level: " << int(level);
    
    QWGHeader header;
    EXPECT_TRUE(readQWGHeader(output_file, header));
    EXPECT_EQ(header.wavelet_levels, level);
    
    EXPECT_GE(header.tile_size, 1u << level);
  }
}

TEST_F(QWGBuilderTest, MemoryOnlyConversion) 
{
  constexpr size_t size = 64;
  auto input_file = createTestGeoTIFF(size, size, [this](size_t, size_t) {
    return height_dist_(rng_);
  });
  
  TerrainConfig config;
  QWGBuilder builder(config);
  
  std::string error_message;
  
  bool success = builder.buildFromGeoTIFF(input_file, "", error_message);
  EXPECT_TRUE(success);
}

}
}