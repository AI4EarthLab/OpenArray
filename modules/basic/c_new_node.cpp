
///:include "../../NodeType.fypp"
#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  ///:for op in [i for i in L if i[3] in ['A','B','F','H']]
  void c_new_node_${op[1]}$(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_${op[1]}$(*u, *v);
  }
  ///:endfor

  ///:for op in [i for i in L if i[3] == 'C']
  void c_new_node_${op[1]}$(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_${op[1]}$(*u);
  }
  ///:endfor
}
