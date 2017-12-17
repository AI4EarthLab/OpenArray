
#include <map>
#include <string>
#include "../Node.hpp"
#include <unordered_map>

typedef std::unordered_map<std::string, NodePtr*> CachedNodes;

CachedNodes cached_nodes;

extern "C"{
  
  void c_find_node(NodePtr*& p, char* key){
    CachedNodes::iterator it;
    it = cached_nodes.find(std::string(key));
    if (it != cached_nodes.end()) {
      p = it->second;
    }else{
      p = NULL;
    }
    return;
  }

  void c_cache_node(NodePtr*& p, char* key){
    CachedNodes::iterator it;    
    it = cached_nodes.find(std::string(key));
    if(it != cached_nodes.end()){
      std::cout<<"Error: found a same key already cached. Maybe hash conflicted?";
    }else{
      cached_nodes[std::string(key)] = p;
    }
  }

  void c_is_null(void*& p, int* i){
    if(p == NULL) {
      *i = 0;
    }else{
      *i = 1;
    }
  }
  
}
