#ifndef VKR_LANDING_LANDING_UNIT_HPP_
#define VKR_LANDING_LANDING_UNIT_HPP_

#include "vkr/landing/types.hpp"
#include "vkr/config.hpp"
#include "vkr/geometry/collision/collision_query.hpp"
#include <memory>

namespace vkr
{
namespace landing
{

class LandingUnit
{
public:
  explicit LandingUnit(const LandingConfig& config);
  ~LandingUnit();


  LandingResult analyzePointCloud(const PointCloud& cloud,
                                 const Eigen::Vector3f& current_position);
  
  void updateConfig(const LandingConfig& config);
  const LandingConfig& getConfig() const;
  
  void exportToJSON(const std::string& filename) const;
  
private:
  struct Implementation;
  std::unique_ptr<Implementation> impl_;
};

}
}

#endif