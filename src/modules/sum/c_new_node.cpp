#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  void c_new_node_csum(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_csum(*u, *v);
  }

  void c_new_node_sum(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_sum(*u, *v);
  }
}