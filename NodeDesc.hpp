
#ifndef __OP_DESC_HPP__
#define __OP_DESC_HPP__

#include "common.hpp"

struct NodeDesc{
  int type; //operator type
  const char* name; //operator name
  bool ew; //if element-wise operation
  bool cl; //if change data layout
  bool expr; // expression form
};

typedef std::vector<NodeDesc> OpDescList;

#endif


