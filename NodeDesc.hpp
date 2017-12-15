
#ifndef __OP_DESC_HPP__
#define __OP_DESC_HPP__

#include "common.hpp"
#include <string>
#include <functional>
#include <vector>

typedef ArrayPtr kernel_rawptr (std::vector<ArrayPtr>&);

typedef std::function<kernel_rawptr> KernelPtr;

//typedef std::function<ArrayPtr(std::vector<ArrayPtr>&)> KernelPtr;

struct NodeDesc{
  NodeType type;         // operator type
  std::string name; // operator name
  std::string sy;   // operator symbol
  bool ew;          // if element-wise operation
  bool cl;          // if change data layout
  std::string expr; // expression form
  KernelPtr func;   // operator function address
  int rt;           // result type

  NodeDesc(){}
  
  // NodeDesc(NodeType _type,
  //          const char* _name,
  //          const char* _sy,
  //          bool _ew, bool _cl,
  //          const char* _expr,
  //          kernel_rawptr* _func,
  //          int _rt) : type(_type), name(_name),
  //                     sy(_sy), ew(_ew), cl(_cl), func(_func), rt(_rt){
    
  // }
};



typedef std::vector<NodeDesc> OpDescList;


#endif

