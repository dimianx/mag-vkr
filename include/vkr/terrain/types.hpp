#ifndef VKR_TERRAIN_TYPES_HPP_
#define VKR_TERRAIN_TYPES_HPP_

namespace vkr
{
namespace terrain
{

struct MinMax
{
  float min;
  float max;
  
  bool isValid() const
  {
    return min <= max;
  }
  
  float range() const
  {
    return max - min;
  }
  
  bool contains(float value) const
  {
    return value >= min && value <= max;
  }
};

}
}

#endif