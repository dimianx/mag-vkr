#ifndef VKR_TERRAIN_QWG_BUILDER_HPP_
#define VKR_TERRAIN_QWG_BUILDER_HPP_

#include "vkr/config.hpp"
#include <string>
#include <memory>
#include <functional>
#include <cstddef>

namespace vkr
{
namespace terrain
{

class QWGBuilder
{
public:
  explicit QWGBuilder(const TerrainConfig& config);
  ~QWGBuilder();
  
  using ProgressCallback = std::function<void(size_t, size_t, const std::string&)>;
  
  bool buildFromGeoTIFF(const std::string& input_file,
                       const std::string& output_file,
                       std::string& error_message,
                       ProgressCallback callback = nullptr);
  
  enum class WaveletType
  {
    HAAR,
  };
  
  void setWaveletType(WaveletType type);
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

}
}

#endif