
#include "../../Function.hpp"
#include "../../Operator.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

extern "C"{
  void c_new_node_sub_node(NodePtr*& A,
          NodePtr*& B, int* ra, int* rb,int* rc){
    if(A == NULL) A = new NodePtr();
    *A = oa::ops::new_node_sub(*B,
            Box(ra[0], ra[1], rb[0], rb[1], rc[0], rc[1]));
  }
  
  void c_new_node_slice_node(NodePtr*& A,
          NodePtr*& B, int* k){
    if(A == NULL) A = new NodePtr();
    *A = oa::ops::new_node_slice(*B, k[0]);
  }
}
