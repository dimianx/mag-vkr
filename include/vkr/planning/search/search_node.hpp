#ifndef VKR_PLANNING_SEARCH_SEARCH_NODE_HPP_
#define VKR_PLANNING_SEARCH_SEARCH_NODE_HPP_

#include "vkr/common.hpp"
#include <Eigen/Core>

namespace vkr
{
namespace planning
{
namespace search
{

struct SearchNode
{
  NodeId id = INVALID_NODE;
  float g_cost = INFINITY_F;
  float rhs = INFINITY_F;
  float h_cost = 0.0f;
  uint32_t back_pointer = INVALID_ID;
  
  uint8_t flags = 0;
  static constexpr uint8_t FLAG_OPEN = 0x01;
  static constexpr uint8_t FLAG_CLOSED = 0x02;
  static constexpr uint8_t FLAG_INCONSISTENT = 0x04;
  static constexpr uint8_t FLAG_IN_CORRIDOR = 0x08;
  
  struct HeuristicSet
  {
    float h0 = 0.0f;
    float h1 = 0.0f;
    float h2 = 0.0f;
    
    float getWeighted(float w0, float w1, float w2) const
    {
      return w0 * h0 + w1 * h1 + w2 * h2;
    }
  } heuristics;
  
  struct Key
  {
    float k1;
    float k2;
    
    bool operator<(const Key& other) const
    {
      return k1 < other.k1 || (k1 == other.k1 && k2 < other.k2);
    }
    
    bool operator>(const Key& other) const
    {
      return other < *this;
    }
    
    bool operator<=(const Key& other) const
    {
      return !(other < *this);
    }
    
    bool operator>=(const Key& other) const
    {
      return !(*this < other);
    }
    
    bool operator==(const Key& other) const
    {
      return k1 == other.k1 && k2 == other.k2;
    }
    
    bool operator!=(const Key& other) const
    {
      return !(*this == other);
    }
  };
  
  Key getKey(float epsilon = 1.0f) const
  {
    float min_g_rhs = std::min(g_cost, rhs);
    return {min_g_rhs + epsilon * h_cost, min_g_rhs};
  }
  
  bool isOpen() const { return flags & FLAG_OPEN; }
  bool isClosed() const { return flags & FLAG_CLOSED; }
  bool isInconsistent() const { return flags & FLAG_INCONSISTENT; }
  
  void setOpen(bool value)
  {
    if (value) flags |= FLAG_OPEN;
    else flags &= ~FLAG_OPEN;
  }
  
  void setClosed(bool value)
  {
    if (value) flags |= FLAG_CLOSED;
    else flags &= ~FLAG_CLOSED;
  }
};

}
}
}

#endif