#ifndef __CACHE_HPP__
#define __CACHE_HPP__

#include <map>
#include <string>
#include "Node.hpp"
#include <unordered_map>

extern "C" {
  void find_node(NodePtr& p, std::string key);
  void cache_node(NodePtr& p, std::string key);
  std::string gen_node_key(const char* file, const int line);
  bool is_valid(NodePtr& p);
  void clear_cache();
}

#endif