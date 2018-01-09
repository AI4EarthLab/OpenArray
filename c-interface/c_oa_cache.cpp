
#include <map>
#include <string>
#include "../Node.hpp"
#include <unordered_map>
#include "../Operator.hpp"

typedef std::unordered_map<std::string, NodePtr> CachedNodes;

CachedNodes cached_nodes;

extern "C"{
  
  void c_find_node(NodePtr*& p, char* key){
    if(p == NULL) p = new NodePtr();
    CachedNodes::iterator it;
    it = cached_nodes.find(std::string(key));
    if (it != cached_nodes.end()) {
      *p = it->second;
      //std::cout<<"found in cache!"<<std::endl;
    }else{
      NodePtr np;
      *p = np;
      //std::cout<<"not found in cache!"<<std::endl;
    }
    return;
  }

  void c_cache_node(NodePtr*& p, char* key){
    CachedNodes::iterator it;    
    it = cached_nodes.find(std::string(key));
    if(it != cached_nodes.end()){
      std::cout<<"Error: found a same key already cached. "
        "Maybe hash conflicted?"<<std::endl;
    }else{
      cached_nodes[std::string(key)] = *p;
      oa::ops::gen_kernels_JIT_with_op(*p);
    }
  }

  void c_is_null(NodePtr*& p, int* i){
    if(p == NULL || *p == NULL) {
      *i = 0;
    }else{
      *i = 1;
    }
  }
  
}
