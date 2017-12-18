
#include "../../Function.hpp"
#include "../../Operator.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

extern "C"{
  void c_new_node_sub_node(NodePtr*& A,
          NodePtr*& B, int* ra, int* rb,int* rc){
    //c_destroy_node((void*&)A);
  
    // ArrayPtr* p = new ArrayPtr();
    // NodePtr &B1 = *((NodePtr*)B);  
    // *p = oa::funcs::subarray(oa::ops::eval(B1),
    //                          Box(ra[0], ra[1],
    //                              rb[0], rb[1],
    //                              rc[0], rc[1]));
    // A = p;

    NodePtr* p  = new NodePtr();
    *p = oa::ops::new_node(*B,
            Box(ra[0], ra[1], rb[0], rb[1], rc[0], rc[1]));
    A = p;
  }

  void c_new_node_sub_array(NodePtr*& A,
          ArrayPtr*& B, int* ra, int* rb,int* rc){

    //c_destroy_node((void*&)A);
    // ArrayPtr* p = new ArrayPtr();

    // *p = oa::funcs::subarray(*(ArrayPtr*)B, Box(ra[0], ra[1],
    //                                             rb[0], rb[1],
    //                                             rc[0], rc[1]));
    // A = p;

    NodePtr* p = new NodePtr();
    
    *p = oa::ops::new_node(*B,
            Box(ra[0], ra[1], rb[0], rb[1], rc[0], rc[1]));
    A = p;
  }
}
