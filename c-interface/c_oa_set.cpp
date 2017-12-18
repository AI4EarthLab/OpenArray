#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Internal.hpp"

extern "C"{

  ///:for type1 in ['int','float','double']
  void c_set_ref_const_${type1}$(NodePtr*& A, ${type1}$ val){
    oa::funcs::set_ref_const((*A)->input(0)->get_data(),
            (*A)->get_ref(), val);
  }

  //set a const value to an array.
  void c_set_array_const_${type1}$(ArrayPtr*& A, ${type1}$ val){
    Shape s = (*A)->buffer_shape();
    int size = s[0] * s[1] * s[3];
    DataType dt = (*A)->get_data_type();

    void* buf = (*A)->get_buffer();
    
    switch(dt){
    case DATA_INT:
      oa::internal::set_buffer_consts((int*)buf,
              size, (int)val);
      break;
    case DATA_FLOAT:
      oa::internal::set_buffer_consts((float*)buf,
              size, (float)val);
      break;
    case DATA_DOUBLE:
      oa::internal::set_buffer_consts((double*)buf,
              size, (double)val);
      break;
    }
  }
  
  ///:endfor

  // void c_set1(void*& A, int* ra, int* rb,int* rc, void*& B){
  //   oa::funcs::set(*(ArrayPtr*)A, Box(ra[0], ra[1],
  //                                     rb[0], rb[1],
  //                                     rc[0], rc[1]),
  //                                     *(ArrayPtr*)B);
  // }

  void c_set_ref_array(NodePtr*& A, ArrayPtr*& B){
    ArrayPtr& p = (*A)->input(0)->get_data();
    oa::funcs::set(p, (*A)->get_ref(), *B);
  }

  void c_set_ref_ref(NodePtr*& A, NodePtr*& B){
    ArrayPtr& p1 = (*A)->input(0)->get_data();
    ArrayPtr& p2 = (*B)->input(0)->get_data();
    
    oa::funcs::set(p1, (*A)->get_ref(), p2, (*B)->get_ref());
  }
  
  // void c_set2(void*& A,
  //         int* ra, int* rb,int* rc, void*& B, int* sa, int* sb,int* sc){
  //   oa::funcs::set(*(ArrayPtr*)A, Box(ra[0], ra[1],
  //                                     rb[0], rb[1],
  //                                     rc[0], rc[1]),
  //                                     *(ArrayPtr*)B,
  //                                 Box(sa[0], sa[1],
  //                                     sb[0], sb[1],
  //                                     sc[0], sc[1]));
  // }


}
