#include "vkr/terrain/qwg_builder.hpp"
#include "vkr/math/wavelets/wavelet_transform_2d.hpp"
#include "vkr/geometry/morton.hpp"
#include "vkr/math/simd/batch_operations.hpp"
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace vkr
{
namespace terrain
{

namespace
{

constexpr uint32_t QWG_MAGIC = 0x47574851;
constexpr uint32_t QWG_VERSION = 1;

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

struct TileInfo
{
  uint32_t morton_code;
  uint32_t data_offset;
  uint32_t compressed_size;
  uint16_t min_height_quantized;
  uint16_t max_height_quantized;
  float coeff_min;
  float coeff_max;
};
#pragma pack(pop)

uint32_t computeOptimalTileSize(size_t page_size, size_t bytes_per_sample, uint32_t wavelet_levels)
{
  uint32_t target_size = static_cast<uint32_t>(std::sqrt(page_size / bytes_per_sample));
  
  uint32_t pow2_size = 1;
  while (pow2_size < target_size)
  {
    pow2_size *= 2;
  }
  if (pow2_size > 1 && (pow2_size - target_size > target_size - pow2_size / 2))
  {
    pow2_size /= 2;
  }
  
  uint32_t min_size = 1u << wavelet_levels;
  pow2_size = std::max(pow2_size, min_size);
  
  constexpr uint32_t MIN_TILE_SIZE = 64;
  constexpr uint32_t MAX_TILE_SIZE = 2048;
  
  return std::min(MAX_TILE_SIZE, std::max(MIN_TILE_SIZE, pow2_size));
}

inline bool isValidValue(float val, float nodata_val)
{
  return std::isfinite(val) && (std::abs(val - nodata_val) > 1e-6f);
}

}

struct QWGBuilder::Implementation
{
  TerrainConfig config;
  WaveletType wavelet_type = WaveletType::HAAR;
  float cell_size = 0.25f;
  
  QWGHeader header = 
  {
  };
  std::vector<TileInfo> tile_infos;
  std::vector<std::vector<uint16_t>> tile_data;
  
  std::unique_ptr<math::wavelets::WaveletTransform2D<math::wavelets::HaarWavelet2D>> haar_transform;
  
  Implementation(const TerrainConfig& cfg) : config(cfg) 
  {
    haar_transform = std::make_unique<math::wavelets::WaveletTransform2D<math::wavelets::HaarWavelet2D>>();
  }
  
  bool processGeoTIFF(const std::string& input_file,
                     const std::string& output_file,
                     std::string& error_message,
                     ProgressCallback callback)
  {
    GDALAllRegister();
    
    std::unique_ptr<GDALDataset, decltype(&GDALClose)> dataset(
        static_cast<GDALDataset*>(GDALOpen(input_file.c_str(), GA_ReadOnly)),
        &GDALClose);
    
    if (!dataset)
    {
      error_message = "Failed to open GeoTIFF file: " + input_file;
      return false;
    }
    
    GDALRasterBand* band = dataset->GetRasterBand(1);
    if (!band)
    {
      error_message = "No raster band found in GeoTIFF";
      return false;
    }
    
    int raster_width = band->GetXSize();
    int raster_height = band->GetYSize();
    double geo_transform[6];
    dataset->GetGeoTransform(geo_transform);
    
    int has_nodata = 0;
    double nodata_value_double = band->GetNoDataValue(&has_nodata);
    if (!has_nodata)
    {
      nodata_value_double = -9999.0;
      spdlog::warn("No NODATA value in raster, using default: {}", nodata_value_double);
    }
    
    float origin_x = static_cast<float>(geo_transform[0]);
    float origin_y = static_cast<float>(geo_transform[3]);
    cell_size = static_cast<float>(std::abs(geo_transform[1]));
    
    spdlog::info("Input raster: {}x{}, cell size: {:.2f}m, nodata: {}", 
                 raster_width, raster_height, cell_size, nodata_value_double);
    
    uint32_t tile_size = computeOptimalTileSize(config.page_size, sizeof(uint16_t), config.wavelet_levels);
    spdlog::info("Computed optimal tile size: {}x{}", tile_size, tile_size);
    
    uint32_t padded_width = ((raster_width + tile_size - 1) / tile_size) * tile_size;
    uint32_t padded_height = ((raster_height + tile_size - 1) / tile_size) * tile_size;
    
    uint32_t tiles_x = padded_width / tile_size;
    uint32_t tiles_y = padded_height / tile_size;
    uint32_t total_tiles = tiles_x * tiles_y;
    
    header = 
    {
    };
    header.magic = QWG_MAGIC;
    header.version = QWG_VERSION;
    header.width = padded_width;
    header.height = padded_height;
    header.tile_size = tile_size;
    header.wavelet_levels = config.wavelet_levels;
    header.origin_x = origin_x;
    header.origin_y = origin_y;
    header.cell_size = cell_size;
    header.num_tiles = total_tiles;
    header.tile_index_offset = sizeof(QWGHeader);
    header.tile_data_offset = sizeof(QWGHeader) + total_tiles * sizeof(TileInfo);
    header.nodata_value = static_cast<float>(nodata_value_double);
    
    switch (wavelet_type)
    {
      case WaveletType::HAAR:
        header.wavelet_type = 0;
        break;
      default:
        header.wavelet_type = 0;
        break;
    }
    
    if (callback)
    {
      callback(0, 100, "Pass 1: Computing global min/max");
    }
    
    if (!findGlobalMinMax(band, raster_width, raster_height, header.nodata_value, 
                         header.global_min_height, header.global_max_height, 
                         error_message, callback))
    {
      return false;
    }
    
    spdlog::info("Global height range: [{:.2f}, {:.2f}]", 
                 header.global_min_height, header.global_max_height);
    
    tile_infos.clear();
    tile_infos.reserve(total_tiles);
    tile_data.clear();
    tile_data.reserve(total_tiles);
    
    if (callback)
    {
      callback(50, 100, "Pass 2: Processing tiles");
    }
    
    if (!processTiles(band, tiles_x, tiles_y, tile_size, error_message, callback))
    {
      return false;
    }
    
    sortTilesByMorton();
    
    recalculateOffsets();
    
    if (!output_file.empty())
    {
      if (!writeToFile(output_file, error_message))
      {
        return false;
      }
    }
    
    if (callback)
    {
      callback(100, 100, "Conversion complete");
    }
    
    spdlog::info("Created QWG with {} tiles", total_tiles);
    
    return true;
  }
  
private:
  bool findGlobalMinMax(GDALRasterBand* band, int width, int height,
                       float nodata_value, float& global_min, float& global_max,
                       std::string& error_message, ProgressCallback callback)
  {
    int success_min, success_max;
    double gdal_min = band->GetMinimum(&success_min);
    double gdal_max = band->GetMaximum(&success_max);
    
    if (success_min && success_max && 
        isValidValue(static_cast<float>(gdal_min), nodata_value) &&
        isValidValue(static_cast<float>(gdal_max), nodata_value))
    {
      spdlog::info("Using pre-computed min/max from raster: [{:.2f}, {:.2f}]", gdal_min, gdal_max);
      global_min = static_cast<float>(gdal_min);
      global_max = static_cast<float>(gdal_max);
      return true;
    }
    
    spdlog::info("Scanning raster manually for global min/max...");
    
    constexpr int SCAN_BLOCK_SIZE = 512;
    std::vector<float> scan_buffer(SCAN_BLOCK_SIZE * SCAN_BLOCK_SIZE);
    
    global_min = std::numeric_limits<float>::max();
    global_max = std::numeric_limits<float>::lowest();
    bool found_valid_data = false;
    
    int total_blocks = ((width + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE) *
                      ((height + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE);
    int processed_blocks = 0;
    
    for (int y = 0; y < height; y += SCAN_BLOCK_SIZE)
    {
      int block_height = std::min(SCAN_BLOCK_SIZE, height - y);
      
      for (int x = 0; x < width; x += SCAN_BLOCK_SIZE)
      {
        int block_width = std::min(SCAN_BLOCK_SIZE, width - x);
        
        CPLErr err = band->RasterIO(GF_Read, x, y, 
                                    block_width, block_height,
                                    scan_buffer.data(), block_width, block_height,
                                    GDT_Float32, 0, 0);
        
        if (err != CE_None)
        {
          error_message = "Failed to read raster block during scan";
          return false;
        }
        
        if (processed_blocks == 0 && block_width > 0 && block_height > 0)
        {
          spdlog::debug("First block sample values: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}]",
                        scan_buffer[0], scan_buffer[1], scan_buffer[2], scan_buffer[3], scan_buffer[4]);
          spdlog::debug("Nodata value: {:.2f}", nodata_value);
          
          int valid_count = 0;
          for (int i = 0; i < std::min(10, block_width * block_height); ++i)
          {
            if (isValidValue(scan_buffer[i], nodata_value))
            {
              valid_count++;
            }
          }
          spdlog::debug("Valid values in first 10 samples: {}/10", valid_count);
        }
        
        for (int i = 0; i < block_width * block_height; ++i)
        {
          if (isValidValue(scan_buffer[i], nodata_value))
          {
            global_min = std::min(global_min, scan_buffer[i]);
            global_max = std::max(global_max, scan_buffer[i]);
            found_valid_data = true;
          }
        }
        
        processed_blocks++;
        if (callback && processed_blocks % 10 == 0)
        {
          int progress = (processed_blocks * 50) / total_blocks;
          callback(progress, 100, "Pass 1: Scanning for min/max");
        }
      }
    }
    
    if (!found_valid_data)
    {
      global_min = 0.0f;
      global_max = 100.0f;
      spdlog::warn("No valid data found in raster, using default range [0, 100]");
    }
    else
    {
      spdlog::info("Found valid data range: [{:.2f}, {:.2f}]", global_min, global_max);
    }
    
    return true;
  }
  
  bool processTiles(GDALRasterBand* band, uint32_t tiles_x, uint32_t tiles_y,
                   uint32_t tile_size, std::string& error_message,
                   ProgressCallback callback)
  {
    std::vector<float> tile_buffer(tile_size * tile_size);
    std::vector<float> read_buffer(tile_size * tile_size);
    
    size_t processed_tiles = 0;
    
    const float nodata_float = header.nodata_value;
    float height_range = header.global_max_height - header.global_min_height;
    float height_quant_step = (height_range > 0) ? height_range / 65535.0f : 1.0f;
    
    spdlog::debug("Processing tiles with nodata={}, global range=[{}, {}]", 
                  nodata_float, header.global_min_height, header.global_max_height);
    
    for (uint32_t ty = 0; ty < tiles_y; ++ty)
    {
      for (uint32_t tx = 0; tx < tiles_x; ++tx)
      {
        if (callback)
        {
          size_t progress = 50 + (processed_tiles * 50) / (tiles_x * tiles_y);
          callback(progress, 100, 
                   "Pass 2: Processing tile " + std::to_string(processed_tiles + 1) + 
                   " of " + std::to_string(tiles_x * tiles_y));
        }
        
        std::fill(tile_buffer.begin(), tile_buffer.end(), nodata_float);
        
        int src_x = tx * tile_size;
        int src_y = ty * tile_size;
        int raster_width = band->GetXSize();
        int raster_height = band->GetYSize();
        int read_width = std::min(static_cast<int>(tile_size), raster_width - src_x);
        int read_height = std::min(static_cast<int>(tile_size), raster_height - src_y);
        
        if (read_width > 0 && read_height > 0)
        {
          CPLErr err = band->RasterIO(GF_Read, src_x, src_y, 
                                      read_width, read_height,
                                      read_buffer.data(), read_width, read_height,
                                      GDT_Float32, 0, 0);
          
          if (err != CE_None)
          {
            error_message = "Failed to read raster data";
            return false;
          }
          
          for (int y = 0; y < read_height; ++y)
          {
            for (int x = 0; x < read_width; ++x)
            {
              tile_buffer[y * tile_size + x] = read_buffer[y * read_width + x];
            }
          }
        }
        
        float tile_min = std::numeric_limits<float>::max();
        float tile_max = std::numeric_limits<float>::lowest();
        float valid_sum = 0.0f;
        int valid_count = 0;
        
        for (const float& val : tile_buffer)
        {
          if (isValidValue(val, nodata_float))
          {
            tile_min = std::min(tile_min, val);
            tile_max = std::max(tile_max, val);
            valid_sum += val;
            valid_count++;
          }
        }
        
        if (valid_count == 0)
        {
          tile_min = header.global_min_height;
          tile_max = header.global_max_height;
          float fill_value = (header.global_min_height + header.global_max_height) * 0.5f;
          std::fill(tile_buffer.begin(), tile_buffer.end(), fill_value);
        }
        else
        {
          float mean_valid = valid_sum / valid_count;
          for (float& val : tile_buffer)
          {
            if (!isValidValue(val, nodata_float))
            {
              val = mean_valid;
            }
          }
        }
        
        tile_min = std::max(tile_min, header.global_min_height);
        tile_max = std::min(tile_max, header.global_max_height);
        
        bool is_uniform = (tile_max - tile_min) < 1e-6f;
        
        if (!is_uniform && wavelet_type == WaveletType::HAAR)
        {
          haar_transform->forward(tile_buffer.data(), tile_size, tile_size, config.wavelet_levels);
        }
        else
        {
          spdlog::debug("Tile ({},{}) has uniform values, skipping wavelet transform", tx, ty);
        }
        
        float coeff_min, coeff_max;
        if (is_uniform)
        {
          coeff_min = coeff_max = tile_buffer[0];
        }
        else
        {
          auto minmax = std::minmax_element(tile_buffer.begin(), tile_buffer.end());
          coeff_min = *minmax.first;
          coeff_max = *minmax.second;
        }
        
        bool has_nan = false;
        for (const float& val : tile_buffer)
        {
          if (!std::isfinite(val))
          {
            has_nan = true;
            break;
          }
        }
        
        if (has_nan || !std::isfinite(coeff_min) || !std::isfinite(coeff_max))
        {
          spdlog::warn("Tile ({},{}) has NaN/Inf coefficients, using original data", tx, ty);
          coeff_min = tile_min;
          coeff_max = tile_max;
          
          float fill_value = (tile_min + tile_max) * 0.5f;
          std::fill(tile_buffer.begin(), tile_buffer.end(), fill_value);
        }
        
        float coeff_range = coeff_max - coeff_min;
        float coeff_scale = (coeff_range > 1e-6f) ? 65535.0f / coeff_range : 1.0f;
        
        std::vector<uint16_t> quantized_coeffs(tile_size * tile_size);
        
        for (size_t i = 0; i < tile_buffer.size(); ++i)
        {
          float normalized = (tile_buffer[i] - coeff_min) * coeff_scale;
          int quantized = std::lround(normalized);
          quantized = std::max(0, std::min(65535, quantized));
          quantized_coeffs[i] = static_cast<uint16_t>(quantized);
        }
        
        TileInfo tile_info;
        tile_info.morton_code = geometry::morton2d::encode(tx, ty);
        tile_info.data_offset = 0;
        tile_info.compressed_size = quantized_coeffs.size() * sizeof(uint16_t);
        tile_info.coeff_min = coeff_min;
        tile_info.coeff_max = coeff_max;
        
        tile_info.min_height_quantized = static_cast<uint16_t>(
            std::max(0.0f, std::min(65535.0f, (tile_min - header.global_min_height) / height_quant_step)));
        tile_info.max_height_quantized = static_cast<uint16_t>(
            std::max(0.0f, std::min(65535.0f, (tile_max - header.global_min_height) / height_quant_step)));
        
        tile_infos.push_back(tile_info);
        tile_data.push_back(std::move(quantized_coeffs));
        
        processed_tiles++;
      }
    }
    
    return true;
  }
  
  void sortTilesByMorton()
  {
    std::vector<size_t> indices(tile_infos.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [this](size_t a, size_t b) 
              {
                return tile_infos[a].morton_code < tile_infos[b].morton_code;
              });
    
    std::vector<TileInfo> sorted_infos;
    std::vector<std::vector<uint16_t>> sorted_data;
    sorted_infos.reserve(tile_infos.size());
    sorted_data.reserve(tile_data.size());
    
    for (size_t idx : indices)
    {
      sorted_infos.push_back(tile_infos[idx]);
      sorted_data.push_back(std::move(tile_data[idx]));
    }
    
    tile_infos = std::move(sorted_infos);
    tile_data = std::move(sorted_data);
  }
  
  void recalculateOffsets()
  {
    uint32_t current_offset = header.tile_data_offset;
    
    for (auto& tile_info : tile_infos)
    {
      tile_info.data_offset = current_offset;
      current_offset += tile_info.compressed_size;
    }
    
    spdlog::debug("Recalculated offsets for {} tiles, total data size: {} bytes", 
                  tile_infos.size(), current_offset - header.tile_data_offset);
  }
  
  bool writeToFile(const std::string& filename, std::string& error_message)
  {
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
      error_message = "Failed to create output file: " + filename;
      return false;
    }
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));
    
    file.write(reinterpret_cast<const char*>(tile_infos.data()),
               tile_infos.size() * sizeof(TileInfo));
    
    for (const auto& data : tile_data)
    {
      file.write(reinterpret_cast<const char*>(data.data()),
                 data.size() * sizeof(uint16_t));
    }
    
    return true;
  }
};

QWGBuilder::QWGBuilder(const TerrainConfig& config)
  : impl_(std::make_unique<Implementation>(config))
{
}

QWGBuilder::~QWGBuilder() = default;

bool QWGBuilder::buildFromGeoTIFF(const std::string& input_file,
                                  const std::string& output_file,
                                  std::string& error_message,
                                  ProgressCallback callback)
{
  return impl_->processGeoTIFF(input_file, output_file, error_message, callback);
}

void QWGBuilder::setWaveletType(WaveletType type)
{
  impl_->wavelet_type = type;
}

}
}