
///:include "../../NodeType.fypp"
#include "../../NodePool.hpp"
#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  ///:for op in [i for i in L if i[3] in ['A','B']]
  void c_new_node_${op[1]}$(NodePtr*& A, int type,
                            NodePtr*& u, NodePtr*& v){
    //c_destroy_node((void*)A);

    // A = new NodePtr();
    // *A = new_node_${op[1]}$(type, u, v);
  }
  ///:endfor

  ///:for op in [i for i in L if i[3] == 'C']
  void c_new_node_${op[1]}$(NodePtr*& A, int type, NodePtr*& u){
    //c_destroy_node((void*)A);

    // A = new NodePtr();
    // *A = new_node_${op[1]}$(type, u);
  }
  ///:endfor
  
}
