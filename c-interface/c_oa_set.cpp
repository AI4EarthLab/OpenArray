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
    int size = s[0] * s[1] * s[2];
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


  ///:for type in ['int', 'float', 'double']
  void c_set_array_farray_${type}$(
      ArrayPtr*& ap, ${type}$*& arr, int* s){
    if(ap == NULL) ap = new ArrayPtr();
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];

    ArrayPtr local =
      oa::funcs::create_local_array<${type}$>(sh, arr);
    
    *ap = oa::funcs::l2g(local);
  }
  ///:endfor

  ///:for type in ['int', 'float', 'double']
  void c_set_ref_farray_${type}$(
      NodePtr*& np, ${type}$*& arr, int* s){
    
    assert((*np)->is_ref_data());

    ArrayPtr& ap = (*np)->get_ref_data();
    Box& ref_box = (*np)->get_ref();

    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    oa::funcs::set<${type}$>(ap, ref_box, arr, sh);
  }
  ///:endfor
  
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
