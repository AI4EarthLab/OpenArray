
#ifndef __C_OA_CACHE_HPP__
#define __C_OA_CACHE_HPP__

#include <map>
#include <string>
#include "../Node.hpp"
#include <unordered_map>

extern "C" {
  void c_find_node(NodePtr*& p, char* key);
  void c_cache_node(NodePtr*& p, char* key);
  void c_is_null(NodePtr*& p, int* i);
  void c_clear_cache() ;
}

#endif
