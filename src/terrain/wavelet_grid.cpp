#include "vkr/terrain/wavelet_grid.hpp"
#include "vkr/math/wavelets/wavelet_transform_2d.hpp"
#include "vkr/geometry/morton.hpp"
#include "vkr/math/simd/batch_operations.hpp"
#include "vkr/math/simd/simd_utils.hpp"
#include <mio/mmap.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <unordered_set>
#include <map>

namespace vkr
{
namespace terrain
{

namespace
{

constexpr uint32_t QWG_MAGIC = 0x47574851;
constexpr uint32_t QWG_VERSION = 1;

enum class WaveletTypeCode : uint32_t
{
  HAAR = 0,
  DAUBECHIES_4 = 1,
  DAUBECHIES_6 = 2
};

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

inline bool isValidValue(float val, float nodata_val) 
{
  return std::isfinite(val) && (std::abs(val - nodata_val) > 1e-5f);
}

}

struct ExtendedPage
{
  uint32_t morton_base;
  std::vector<uint16_t> coefficients;
  MinMax height_range;
  std::atomic<uint32_t> access_count
  {
    0
  };
  uint64_t last_access_time;
  
  mutable math::simd::AlignedPtr<float> decompressed_heights;
  mutable std::mutex decompress_mutex;
  mutable std::multimap<uint64_t, uint32_t>::iterator lru_iterator;
  
  float coeff_min;
  float coeff_max;
};

struct ExtendedPageCache
{
  std::unordered_map<uint32_t, std::unique_ptr<ExtendedPage>> pages;
  mutable std::mutex mutex;
  size_t max_pages;
  std::multimap<uint64_t, uint32_t> lru_index;
  size_t current_memory_bytes = 0;
  
  ExtendedPage* getPage(uint32_t morton_code)
  {
    std::lock_guard<std::mutex> lock(mutex);
    
    auto it = pages.find(morton_code);
    if (it != pages.end())
    {
      ExtendedPage* page = it->second.get();
      
      page->access_count.fetch_add(1, std::memory_order_relaxed);
      
      if (page->lru_iterator != lru_index.end())
      {
        lru_index.erase(page->lru_iterator);
      }
      
      uint64_t new_time = std::chrono::steady_clock::now().time_since_epoch().count();
      page->last_access_time = new_time;
      page->lru_iterator = lru_index.emplace(new_time, morton_code);
      
      return page;
    }
    
    return nullptr;
  }
  
  void evictLRU()
  {
    if (pages.empty() || lru_index.empty()) 
    {
      return;
    }
    
    auto oldest = lru_index.begin();
    uint32_t lru_morton = oldest->second;
    lru_index.erase(oldest);
    
    auto page_it = pages.find(lru_morton);
    if (page_it != pages.end())
    {
      ExtendedPage* page = page_it->second.get();
      size_t page_memory = sizeof(ExtendedPage) + 
                          page->coefficients.size() * sizeof(uint16_t);
      if (page->decompressed_heights)
      {
        page_memory += page->coefficients.size() * sizeof(float);
      }
      current_memory_bytes -= page_memory;
    }
    
    pages.erase(lru_morton);
  }
  
  void addPage(uint32_t morton_code, std::unique_ptr<ExtendedPage> page)
  {
    if (pages.count(morton_code) > 0)
    {
      return;
    }
    
    ExtendedPage* ext_page = page.get();
    
    size_t page_memory = sizeof(ExtendedPage) + ext_page->coefficients.size() * sizeof(uint16_t);
    
    size_t max_memory = max_pages * 1024 * 1024;
    while (current_memory_bytes + page_memory > max_memory && !pages.empty())
    {
      evictLRU();
    }
    
    ext_page->lru_iterator = lru_index.emplace(ext_page->last_access_time, morton_code);
    current_memory_bytes += page_memory;
    pages[morton_code] = std::move(page);
  }
};

struct WaveletGrid::Implementation
{
  TerrainConfig config;
  
  mio::mmap_source mmap;
  
  QWGHeader header;
  std::vector<TileInfo> tile_infos;
  
  mutable ExtendedPageCache cache;
  
  std::unique_ptr<math::wavelets::WaveletTransform2D<math::wavelets::HaarWavelet2D>> haar_transform;
  
  enum class ActiveTransform 
  { 
    HAAR, DAUBECHIES 
  } active_transform = ActiveTransform::HAAR;
  
  uint32_t tiles_per_row;
  uint32_t tiles_per_col;
  float inv_cell_size;
  
  Implementation(const TerrainConfig& cfg) 
    : config(cfg)
  {
    cache.max_pages = config.max_cache_size;
    
    haar_transform = std::make_unique<math::wavelets::WaveletTransform2D<math::wavelets::HaarWavelet2D>>();
  }
  
  bool loadFile(const std::string& filename)
  {
    std::error_code error;
    mmap = mio::make_mmap_source(filename, 0, mio::map_entire_file, error);
    
    if (error)
    {
      spdlog::error("Failed to mmap terrain file: {} - {}", filename, error.message());
      return false;
    }
    
    if (mmap.size() < sizeof(QWGHeader))
    {
      spdlog::error("Terrain file too small");
      return false;
    }
    
    std::memcpy(&header, mmap.data(), sizeof(QWGHeader));
    
    if (header.magic != QWG_MAGIC)
    {
      spdlog::error("Invalid terrain file magic number");
      return false;
    }
    
    if (header.version != QWG_VERSION)
    {
      spdlog::error("Unsupported terrain file version: {}", header.version);
      return false;
    }
    
    WaveletTypeCode wavelet_code = static_cast<WaveletTypeCode>(header.wavelet_type);
    switch (wavelet_code)
    {
      case WaveletTypeCode::HAAR:
        active_transform = ActiveTransform::HAAR;
        spdlog::info("Using Haar wavelet transform");
        break;
        
      default:
        spdlog::error("Unknown wavelet type code: {}", header.wavelet_type);
        return false;
    }
    
    tile_infos.resize(header.num_tiles);
    std::memcpy(tile_infos.data(), mmap.data() + header.tile_index_offset,
                header.num_tiles * sizeof(TileInfo));
    
    tiles_per_row = header.width / header.tile_size;
    tiles_per_col = header.height / header.tile_size;
    inv_cell_size = 1.0f / header.cell_size;
    
    config.wavelet_levels = header.wavelet_levels;
    
    spdlog::info("Loaded terrain: {}x{} pixels, {} tiles, {:.2f}m resolution, wavelet type {}",
                 header.width, header.height, header.num_tiles, header.cell_size, header.wavelet_type);
    
    return true;
  }
  
  const TileInfo* findTile(uint32_t morton_code) const
  {
    auto it = std::lower_bound(tile_infos.begin(), tile_infos.end(), morton_code,
                              [](const TileInfo& a, uint32_t b) 
                              { 
                                return a.morton_code < b; 
                              });
    
    if (it != tile_infos.end() && it->morton_code == morton_code)
    {
      return &(*it);
    }
    return nullptr;
  }
  
  std::unique_ptr<ExtendedPage> loadTile(const TileInfo& tile) const
  {
    auto page = std::make_unique<ExtendedPage>();
    page->morton_base = tile.morton_code;
    
    float height_range = header.global_max_height - header.global_min_height;
    float height_quant_step = (height_range > 0) ? height_range / 65535.0f : 1.0f;
    page->height_range.min = tile.min_height_quantized * height_quant_step + header.global_min_height;
    page->height_range.max = tile.max_height_quantized * height_quant_step + header.global_min_height;
    
    const uint16_t* tile_data = reinterpret_cast<const uint16_t*>(mmap.data() + tile.data_offset);
    size_t num_coeffs = header.tile_size * header.tile_size;
    
    page->coefficients.resize(num_coeffs);
    std::copy(tile_data, tile_data + num_coeffs, page->coefficients.begin());
    
    page->coeff_min = tile.coeff_min;
    page->coeff_max = tile.coeff_max;
    
    page->last_access_time = std::chrono::steady_clock::now().time_since_epoch().count();
    page->lru_iterator = cache.lru_index.end();
    
    return page;
  }
  
  const float* getDecompressedHeights(const ExtendedPage& page) const
  {
    if (page.decompressed_heights)
    {
      return page.decompressed_heights.get();
    }
    
    std::lock_guard<std::mutex> lock(page.decompress_mutex);
    
    if (page.decompressed_heights)
    {
      return page.decompressed_heights.get();
    }
    
    size_t num_coeffs = header.tile_size * header.tile_size;
    page.decompressed_heights = math::simd::allocateAligned<float>(num_coeffs);
    float* heights = page.decompressed_heights.get();
    
    float coeff_range = page.coeff_max - page.coeff_min;
    float coeff_scale = (coeff_range > 0) ? coeff_range / 65535.0f : 1.0f;
    
    for (size_t i = 0; i < num_coeffs; ++i)
    {
      heights[i] = page.coefficients[i] * coeff_scale + page.coeff_min;
    }
    
    if (active_transform == ActiveTransform::HAAR)
    {
      haar_transform->inverse(heights, header.tile_size, header.tile_size, header.wavelet_levels);
    }
    
    return heights;
  }
  
  ExtendedPage* getOrLoadPage(uint32_t morton_code) const
  {
    ExtendedPage* page = cache.getPage(morton_code);
    if (page)
    {
      return page;
    }
    
    static std::mutex load_mutex;
    std::lock_guard<std::mutex> lock(load_mutex);
    
    page = cache.getPage(morton_code);
    if (page)
    {
      return page;
    }
    
    const TileInfo* tile = findTile(morton_code);
    if (!tile)
    {
      return nullptr;
    }
    
    auto new_page = loadTile(*tile);
    ExtendedPage* page_ptr = new_page.get();
    
    {
      std::lock_guard<std::mutex> cache_lock(cache.mutex);
      cache.addPage(morton_code, std::move(new_page));
    }
    
    return page_ptr;
  }
  
  float getHeightAtPixel(int x, int y) const
  {
    int tile_x = x / header.tile_size;
    int tile_y = y / header.tile_size;
    
    int local_x = x % header.tile_size;
    int local_y = y % header.tile_size;
    
    uint32_t morton = geometry::morton2d::encode(static_cast<uint32_t>(tile_x), static_cast<uint32_t>(tile_y));
    
    ExtendedPage* page = getOrLoadPage(morton);
    if (!page)
    {
      return header.nodata_value;
    }
    
    const float* heights = getDecompressedHeights(*page);
    
    return heights[local_y * header.tile_size + local_x];
  }
};

WaveletGrid::WaveletGrid(const TerrainConfig& config)
  : impl_(std::make_unique<Implementation>(config))
{
}

WaveletGrid::~WaveletGrid() = default;

WaveletGrid::WaveletGrid(WaveletGrid&&) noexcept = default;
WaveletGrid& WaveletGrid::operator=(WaveletGrid&&) noexcept = default;

bool WaveletGrid::loadFromFile(const std::string& filename)
{
  return impl_->loadFile(filename);
}

float WaveletGrid::getHeight(float x, float y) const
{
  float grid_x = (x - impl_->header.origin_x) * impl_->inv_cell_size;
  float grid_y = (impl_->header.origin_y - y) * impl_->inv_cell_size;
  
  if (grid_x < 0 || grid_y < 0 || 
      grid_x >= impl_->header.width - 1 || 
      grid_y >= impl_->header.height - 1)
  {
    return 0.0f;
  }
  
  int ix = static_cast<int>(grid_x);
  int iy = static_cast<int>(grid_y);
  
  float fx = grid_x - ix;
  float fy = grid_y - iy;
  
  float h00 = impl_->getHeightAtPixel(ix, iy);
  float h10 = impl_->getHeightAtPixel(ix + 1, iy);
  float h01 = impl_->getHeightAtPixel(ix, iy + 1);
  float h11 = impl_->getHeightAtPixel(ix + 1, iy + 1);
  
  float nodata = impl_->header.nodata_value;
  constexpr float NODATA_EPSILON = 1e-5f;
  bool is_nodata[4] = 
  {
    std::abs(h00 - nodata) < NODATA_EPSILON,
    std::abs(h10 - nodata) < NODATA_EPSILON,
    std::abs(h01 - nodata) < NODATA_EPSILON,
    std::abs(h11 - nodata) < NODATA_EPSILON
  };
  
  if (is_nodata[0] || is_nodata[1] || is_nodata[2] || is_nodata[3])
  {
    float best_height = nodata;
    float best_dist = std::numeric_limits<float>::max();
    
    if (!is_nodata[0]) 
    {
      float dist = fx * fx + fy * fy;
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h00; 
      }
    }
    if (!is_nodata[1]) 
    {
      float dist = (1-fx) * (1-fx) + fy * fy;
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h10; 
      }
    }
    if (!is_nodata[2]) 
    {
      float dist = fx * fx + (1-fy) * (1-fy);
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h01; 
      }
    }
    if (!is_nodata[3]) 
    {
      float dist = (1-fx) * (1-fx) + (1-fy) * (1-fy);
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h11; 
      }
    }
    
    return best_height;
  }
  
  float h0 = h00 * (1 - fx) + h10 * fx;
  float h1 = h01 * (1 - fx) + h11 * fx;
  return h0 * (1 - fy) + h1 * fy;
}

MinMax WaveletGrid::getMinMax(const geometry::BoundingBox& bbox) const
{
  MinMax range
  {
    std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()
  };

  float grid_min_x_f = (bbox.min.x() - impl_->header.origin_x) * impl_->inv_cell_size;
  float grid_min_y_f = (impl_->header.origin_y - bbox.max.y()) * impl_->inv_cell_size;
  float grid_max_x_f = (bbox.max.x() - impl_->header.origin_x) * impl_->inv_cell_size;
  float grid_max_y_f = (impl_->header.origin_y - bbox.min.y()) * impl_->inv_cell_size;

  int start_px = std::max(0, static_cast<int>(std::floor(grid_min_x_f)));
  int start_py = std::max(0, static_cast<int>(std::floor(grid_min_y_f)));
  int end_px = std::min(static_cast<int>(impl_->header.width), static_cast<int>(std::ceil(grid_max_x_f)));
  int end_py = std::min(static_cast<int>(impl_->header.height), static_cast<int>(std::ceil(grid_max_y_f)));

  if (start_px >= end_px || start_py >= end_py) 
  {
    return range;
  }

  const uint32_t tile_size = impl_->header.tile_size;
  int start_tx = start_px / tile_size;
  int start_ty = start_py / tile_size;
  int end_tx = (end_px - 1) / tile_size;
  int end_ty = (end_py - 1) / tile_size;

  for (int ty = start_ty; ty <= end_ty; ++ty) 
  {
    for (int tx = start_tx; tx <= end_tx; ++tx) 
    {
      uint32_t morton = geometry::morton2d::encode(static_cast<uint32_t>(tx), static_cast<uint32_t>(ty));
      ExtendedPage* page = impl_->getOrLoadPage(morton);
      if (!page) continue;

      int tile_start_px = tx * tile_size;
      int tile_start_py = ty * tile_size;
      int tile_end_px = tile_start_px + tile_size;
      int tile_end_py = tile_start_py + tile_size;

      if (start_px <= tile_start_px && end_px >= tile_end_px &&
          start_py <= tile_start_py && end_py >= tile_end_py)
      {
        range.min = std::min(range.min, page->height_range.min);
        range.max = std::max(range.max, page->height_range.max);
      }
      else
      {
        const float* heights = impl_->getDecompressedHeights(*page);
        if (!heights) continue;

        int local_start_x = std::max(start_px, tile_start_px) - tile_start_px;
        int local_start_y = std::max(start_py, tile_start_py) - tile_start_py;
        int local_end_x = std::min(end_px, tile_end_px) - tile_start_px;
        int local_end_y = std::min(end_py, tile_end_py) - tile_start_py;

        for (int ly = local_start_y; ly < local_end_y; ++ly) 
        {
          for (int lx = local_start_x; lx < local_end_x; ++lx) 
          {
            float height = heights[ly * tile_size + lx];
            if (isValidValue(height, impl_->header.nodata_value)) 
            {
              range.min = std::min(range.min, height);
              range.max = std::max(range.max, height);
            }
          }
        }
      }
    }
  }

  return range;
}


void WaveletGrid::batchGetHeight(const float* x, const float* y, size_t count, float* heights) const
{
  using namespace math::simd;
  
  constexpr size_t BATCH_SIZE = 64;
  
  for (size_t batch_start = 0; batch_start < count; batch_start += BATCH_SIZE)
  {
    size_t batch_end = std::min(batch_start + BATCH_SIZE, count);
    
    std::unordered_set<uint32_t> morton_codes;
    for (size_t j = batch_start; j < batch_end; ++j)
    {
      morton_codes.insert(getMortonCode(x[j], y[j]));
    }
    
    for (uint32_t morton : morton_codes)
    {
      loadPage(morton);
    }
    
    simdLoop(batch_end - batch_start, [&](size_t offset, bool is_simd) 
    {
      size_t idx = batch_start + offset;
      
      if (is_simd && idx + SIMD_WIDTH <= batch_end)
      {
        batch_type x_batch = batch_type::load_unaligned(&x[idx]);
        batch_type y_batch = batch_type::load_unaligned(&y[idx]);
        
        batch_type grid_x = (x_batch - impl_->header.origin_x) * impl_->inv_cell_size;
        batch_type grid_y = (impl_->header.origin_y - y_batch) * impl_->inv_cell_size;
        
        alignas(SIMD_ALIGNMENT) float results[SIMD_WIDTH];
        
        alignas(SIMD_ALIGNMENT) float gx_arr[SIMD_WIDTH];
        alignas(SIMD_ALIGNMENT) float gy_arr[SIMD_WIDTH];
        grid_x.store_aligned(gx_arr);
        grid_y.store_aligned(gy_arr);
        
        for (size_t k = 0; k < SIMD_WIDTH; ++k)
        {
          float gx = gx_arr[k];
          float gy = gy_arr[k];
          
          if (gx < 0 || gy < 0 || 
              gx >= impl_->header.width - 1 || 
              gy >= impl_->header.height - 1)
          {
            results[k] = 0.0f;
          }
          else
          {
            int ix = static_cast<int>(gx);
            int iy = static_cast<int>(gy);
            float fx = gx - ix;
            float fy = gy - iy;
            
            float h00 = impl_->getHeightAtPixel(ix, iy);
            float h10 = impl_->getHeightAtPixel(ix + 1, iy);
            float h01 = impl_->getHeightAtPixel(ix, iy + 1);
            float h11 = impl_->getHeightAtPixel(ix + 1, iy + 1);
            
            float h0 = h00 * (1 - fx) + h10 * fx;
            float h1 = h01 * (1 - fx) + h11 * fx;
            results[k] = h0 * (1 - fy) + h1 * fy;
          }
        }
        
        batch_type::load_aligned(results).store_unaligned(&heights[idx]);
      }
      else
      {
        heights[idx] = getHeight(x[idx], y[idx]);
      }
    });
  }
}

geometry::BoundingBox WaveletGrid::getBounds() const
{
  geometry::BoundingBox bounds;
  bounds.min.x() = impl_->header.origin_x;
  bounds.min.y() = impl_->header.origin_y - impl_->header.height * impl_->header.cell_size;
  bounds.min.z() = impl_->header.global_min_height;
  bounds.max.x() = impl_->header.origin_x + impl_->header.width * impl_->header.cell_size;
  bounds.max.y() = impl_->header.origin_y;
  bounds.max.z() = impl_->header.global_max_height;
  return bounds;
}

void WaveletGrid::exportToJSON(const std::string& filename) const
{
  using json = nlohmann::json;
  
  json output;
  
  output["metadata"] = 
  {
    {
      "width", impl_->header.width
    },
    {
      "height", impl_->header.height
    },
    {
      "tile_size", impl_->header.tile_size
    },
    {
      "num_tiles", impl_->header.num_tiles
    },
    {
      "wavelet_levels", impl_->header.wavelet_levels
    },
    {
      "wavelet_type", impl_->header.wavelet_type
    },
    {
      "wavelet_type_name", 
      impl_->header.wavelet_type == 0 ? "HAAR" :
      impl_->header.wavelet_type == 1 ? "DAUBECHIES_4" :
      impl_->header.wavelet_type == 2 ? "DAUBECHIES_6" : "UNKNOWN"
    },
    {
      "origin_x", impl_->header.origin_x
    },
    {
      "origin_y", impl_->header.origin_y
    },
    {
      "cell_size", impl_->header.cell_size
    },
    {
      "global_min_height", impl_->header.global_min_height
    },
    {
      "global_max_height", impl_->header.global_max_height
    },
    {
      "nodata_value", impl_->header.nodata_value
    }
  };
  
  geometry::BoundingBox bounds = getBounds();
  output["bounds"] = 
  {
    {
      "min", 
      {
        bounds.min.x(), bounds.min.y(), bounds.min.z()
      }
    },
    {
      "max", 
      {
        bounds.max.x(), bounds.max.y(), bounds.max.z()
      }
    }
  };
  
  json tiles_json = json::array();
  float height_range = impl_->header.global_max_height - impl_->header.global_min_height;
  float height_quant_step = (height_range > 0) ? height_range / 65535.0f : 1.0f;
  
  for (const auto& tile : impl_->tile_infos)
  {
    uint32_t tx, ty;
    geometry::morton2d::decode(tile.morton_code, tx, ty);
    
    json tile_json = 
    {
      {
        "morton_code", tile.morton_code
      },
      {
        "tile_x", tx
      },
      {
        "tile_y", ty
      },
      {
        "data_offset", tile.data_offset
      },
      {
        "compressed_size", tile.compressed_size
      },
      {
        "min_height", tile.min_height_quantized * height_quant_step + impl_->header.global_min_height
      },
      {
        "max_height", tile.max_height_quantized * height_quant_step + impl_->header.global_min_height
      }
    };
    
    tiles_json.push_back(tile_json);
  }
  output["tiles"] = tiles_json;
  
  output["cache"] = 
  {
    {
      "max_pages", impl_->cache.max_pages
    },
    {
      "current_pages", impl_->cache.pages.size()
    },
    {
      "current_memory_mb", impl_->cache.current_memory_bytes / (1024.0 * 1024.0)
    }
  };
  
  std::ofstream file(filename);
  if (!file)
  {
    spdlog::error("Failed to open file for JSON export: {}", filename);
    return;
  }
  
  file << output.dump(2);
  file.close();
  
  spdlog::info("Exported terrain metadata to JSON: {}", filename);
}

uint32_t WaveletGrid::getMortonCode(float x, float y) const
{
  float grid_x = (x - impl_->header.origin_x) * impl_->inv_cell_size;
  float grid_y = (impl_->header.origin_y - y) * impl_->inv_cell_size;
  
  int tile_x = static_cast<int>(grid_x) / impl_->header.tile_size;
  int tile_y = static_cast<int>(grid_y) / impl_->header.tile_size;
  
  return geometry::morton2d::encode(static_cast<uint32_t>(tile_x), static_cast<uint32_t>(tile_y));
}

float WaveletGrid::interpolateBilinear(float x, float y, uint8_t level) const
{
  float grid_x = (x - impl_->header.origin_x) * impl_->inv_cell_size;
  float grid_y = (impl_->header.origin_y - y) * impl_->inv_cell_size;
  
  float scale = static_cast<float>(1 << level);
  float scaled_x = grid_x / scale;
  float scaled_y = grid_y / scale;
  
  float scaled_width = impl_->header.width / scale;
  float scaled_height = impl_->header.height / scale;
  
  if (scaled_x < 0 || scaled_y < 0 || 
      scaled_x >= scaled_width - 1 || 
      scaled_y >= scaled_height - 1)
  {
    return 0.0f;
  }
  
  int ix = static_cast<int>(scaled_x);
  int iy = static_cast<int>(scaled_y);
  
  float fx = scaled_x - ix;
  float fy = scaled_y - iy;
  
  int orig_x0 = ix * static_cast<int>(scale);
  int orig_y0 = iy * static_cast<int>(scale);
  int orig_x1 = (ix + 1) * static_cast<int>(scale);
  int orig_y1 = (iy + 1) * static_cast<int>(scale);
  
  orig_x1 = std::min(orig_x1, static_cast<int>(impl_->header.width - 1));
  orig_y1 = std::min(orig_y1, static_cast<int>(impl_->header.height - 1));
  
  float h00 = impl_->getHeightAtPixel(orig_x0, orig_y0);
  float h10 = impl_->getHeightAtPixel(orig_x1, orig_y0);
  float h01 = impl_->getHeightAtPixel(orig_x0, orig_y1);
  float h11 = impl_->getHeightAtPixel(orig_x1, orig_y1);
  
  float nodata = impl_->header.nodata_value;
  constexpr float NODATA_EPSILON = 1e-5f;
  bool is_nodata[4] = 
  {
    std::abs(h00 - nodata) < NODATA_EPSILON,
    std::abs(h10 - nodata) < NODATA_EPSILON,
    std::abs(h01 - nodata) < NODATA_EPSILON,
    std::abs(h11 - nodata) < NODATA_EPSILON
  };
  
  if (is_nodata[0] || is_nodata[1] || is_nodata[2] || is_nodata[3])
  {
    float best_height = nodata;
    float best_dist = std::numeric_limits<float>::max();
    
    if (!is_nodata[0]) 
    {
      float dist = fx * fx + fy * fy;
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h00; 
      }
    }
    if (!is_nodata[1]) 
    {
      float dist = (1-fx) * (1-fx) + fy * fy;
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h10; 
      }
    }
    if (!is_nodata[2]) 
    {
      float dist = fx * fx + (1-fy) * (1-fy);
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h01; 
      }
    }
    if (!is_nodata[3]) 
    {
      float dist = (1-fx) * (1-fx) + (1-fy) * (1-fy);
      if (dist < best_dist) 
      { 
        best_dist = dist; 
        best_height = h11; 
      }
    }
    
    return best_height;
  }
  
  float h0 = h00 * (1 - fx) + h10 * fx;
  float h1 = h01 * (1 - fx) + h11 * fx;
  return h0 * (1 - fy) + h1 * fy;
}

void WaveletGrid::loadPage(uint32_t morton_code) const
{
  impl_->getOrLoadPage(morton_code);
}

}
}