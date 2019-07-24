/*
 * NodeDesc.hpp
 * information of each Node
 *
=======================================================*/

#ifndef __OP_DESC_HPP__
#define __OP_DESC_HPP__

#include "common.hpp"
#include <string>
#include <functional>
#include <vector>

// define kernel raw pointer
typedef ArrayPtr kernel_rawptr (std::vector<ArrayPtr>&);

// define kernel function pointer
typedef std::function<kernel_rawptr> KernelPtr;

//typedef std::function<ArrayPtr(std::vector<ArrayPtr>&)> KernelPtr;

struct NodeDesc{
  NodeType type;    // operator type
  std::string name; // operator name
  std::string sy;   // operator symbol
  bool ew;          // is element-wise operation or not
  bool cl;          // change data layout or not
  std::string expr; // expression form
  KernelPtr func;   // operator function address
  int rt;           // result type

  NodeDesc(){}
  
};

typedef std::vector<NodeDesc> OpDescList;

#endif

