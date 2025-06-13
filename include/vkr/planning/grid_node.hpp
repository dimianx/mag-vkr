#ifndef VKR_PLANNING_GRID_NODE_HPP_
#define VKR_PLANNING_GRID_NODE_HPP_

#include "vkr/common.hpp"
#include <cstdint>

namespace vkr
{
namespace planning
{

struct GridNode
{
  uint64_t morton_code = 0;
  NodeId parent = INVALID_NODE;
  float cost_to_here = INFINITY_F;
  uint8_t level = 0;
  
  struct Flags
  {
    uint8_t is_open : 1;
    uint8_t is_closed : 1;
    uint8_t corridor_state : 2;
    uint8_t reserved : 4;
  } flags{};
  
  bool isOpen() const { return flags.is_open; }
  bool isClosed() const { return flags.is_closed; }
  void setOpen(bool value) { flags.is_open = value ? 1 : 0; }
  void setClosed(bool value) { flags.is_closed = value ? 1 : 0; }
};

}
}

#endif