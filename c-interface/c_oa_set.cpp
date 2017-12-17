#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"

extern "C"{
/*
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
*/


  ///:for type1 in ['int','float','double']
  void c_set_with_const_${type1}$(void*& A, int* ra, int* rb,int* rc, ${type1}$* val ){
    oa::funcs::set_with_const(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                                 rb[0], rb[1],
                                                 rc[0], rc[1]), 
                                                 *val);
  }
  ///:endfor
/*
  void c_set_with_const_float(void*& A, int* ra, int* rb,int* rc, float* val ){
    oa::funcs::set_with_const(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                                 rb[0], rb[1],
                                                 rc[0], rc[1]), 
                                                 *val);
  }

  void c_set_with_const_double(void*& A, int* ra, int* rb,int* rc, double* val ){
    oa::funcs::set_with_const(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                                 rb[0], rb[1],
                                                 rc[0], rc[1]), 
                                                 *val);
  }
  */
}
