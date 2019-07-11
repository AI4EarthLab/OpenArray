#include "cache.hpp"
#include <map>
#include <string>
#include "Node.hpp"
#include "Operator.hpp"
#include <unordered_map>

typedef std::unordered_map<std::string, NodePtr> Cached_Nodes;

Cached_Nodes _cached_nodes;

extern "C"{
  
  void find_node(NodePtr& p, std::string key) {
    Cached_Nodes::iterator it;
    it = _cached_nodes.find(key);
    if (it != _cached_nodes.end()) {
      p = it->second;
    }else{
      NodePtr np;
      p = np;
    }
    return;
  }

  void cache_node(NodePtr& p, std::string key) {
    Cached_Nodes::iterator it;    
    it = _cached_nodes.find(key);
    if(it != _cached_nodes.end()){
      std::cout<<"Error: found a same key already cached. Maybe hash conflicted?";
    }else{
      _cached_nodes[key] = p;
      oa::ops::gen_kernels_JIT_with_op(p);
    }
  }
  
  std::string gen_node_key(const char* file, const int line) {
    string tmp_node_key = "";
    tmp_node_key = file;
    tmp_node_key += ":";
    int templine = line;
    tmp_node_key += to_string((long long)templine);
    std::cout<<tmp_node_key<<endl;
    return tmp_node_key;
  }

  bool is_valid(NodePtr& p) {
    if (p == NULL) printf("not found node\n");
    else printf("found node\n");
    return (p != NULL);
  }

  void clear_cache() {
/*
    for (auto it = _cached_nodes.begin(); it != _cached_nodes.end(); it++) {
      cout<<it->first<<" "<<it->second<<endl;
    }
*/
    _cached_nodes.clear();
  }

}
