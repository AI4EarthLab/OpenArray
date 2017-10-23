
#ifndef __OP_DESC_HPP__
#define __OP_DESC_HPP__

#include "common.hpp"
#include <string>

struct NodeDesc{
  int type; //operator type
  std::string name; //operator name
  bool ew; //if element-wise operation
  bool cl; //if change data layout
  std::string expr; // expression form
};

typedef std::vector<NodeDesc> OpDescList;

#endif


