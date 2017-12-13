
#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"

extern "C"{
  void c_sub_node(void*& A, void*& B, int* ra, int* rb,int* rc){
    c_destroy_array(A);
  
    ArrayPtr* p = new ArrayPtr();
    NodePtr &B1 = *((NodePtr*)B);  
    *p = oa::funcs::subarray(oa::ops::eval(B1),
                             Box(ra[0], ra[1],
                                 rb[0], rb[1],
                                 rc[0], rc[1]));
    A = p;
  }

  void c_sub_array(void*& A, void*& B, int* ra, int* rb,int* rc){
    c_destroy_array(A);
    ArrayPtr* p = new ArrayPtr();

    *p = oa::funcs::subarray(*(ArrayPtr*)B, Box(ra[0], ra[1],
                                                rb[0], rb[1],
                                                rc[0], rc[1]));
    A = p;  
  }
}
