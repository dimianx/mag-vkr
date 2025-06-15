#ifndef VKR_TERRAIN_WAVELET_GRID_HPP_
#define VKR_TERRAIN_WAVELET_GRID_HPP_

#include "vkr/terrain/types.hpp"
#include "vkr/geometry/primitives.hpp"
#include "vkr/config.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace vkr
{
namespace terrain
{

class WaveletGrid
{
public:
  explicit WaveletGrid(const TerrainConfig& config);
  ~WaveletGrid();
  
  WaveletGrid(const WaveletGrid&) = delete;
  WaveletGrid& operator=(const WaveletGrid&) = delete;
  
  WaveletGrid(WaveletGrid&&) noexcept;
  WaveletGrid& operator=(WaveletGrid&&) noexcept;
  
  bool loadFromFile(const std::string& filename);
  
  float getHeight(float x, float y) const;
  
  MinMax getMinMax(const geometry::BoundingBox& bbox) const;
  
  void batchGetHeight(const float* x, const float* y, size_t count, float* heights) const;
  
  geometry::BoundingBox getBounds() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct Page
  {
    uint32_t morton_base;
    std::vector<uint16_t> coefficients;
    MinMax height_range;
    std::atomic<uint32_t> access_count{0};
    uint64_t last_access_time;
  };
  
  struct PageCache
  {
    std::unordered_map<uint32_t, std::unique_ptr<Page>> pages;
    mutable std::mutex mutex;
    size_t max_pages;
    
    Page* getPage(uint32_t morton_code);
    void evictLRU();
  };
  
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
  
  uint32_t getMortonCode(float x, float y) const;
  float interpolateBilinear(float x, float y, uint8_t level) const;
  void loadPage(uint32_t morton_code) const;
};

}
}

#endif