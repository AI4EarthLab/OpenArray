#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Internal.hpp"
#include "../TreeRootDict.hpp"

extern "C"{

  //set a const value to an array.
  void c_set_with_mask_array_const_int_array(ArrayPtr*& A,
          int val, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    int sw = (*A)->get_partition()->get_stencil_width();

    ArrayPtr B = oa::funcs::get_seq_scalar(val);

    oa::funcs::set_with_mask((*A), B, (*mask));

  }
  //set a const value to an array.
  void c_set_with_mask_array_const_float_array(ArrayPtr*& A,
          float val, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    int sw = (*A)->get_partition()->get_stencil_width();

    ArrayPtr B = oa::funcs::get_seq_scalar(val);

    oa::funcs::set_with_mask((*A), B, (*mask));

  }
  //set a const value to an array.
  void c_set_with_mask_array_const_double_array(ArrayPtr*& A,
          double val, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    int sw = (*A)->get_partition()->get_stencil_width();

    ArrayPtr B = oa::funcs::get_seq_scalar(val);

    oa::funcs::set_with_mask((*A), B, (*mask));

  }

  void c_set_with_mask_array_array_array(ArrayPtr*& A,
          ArrayPtr*& B, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(B != NULL && *B != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    oa::funcs::set_with_mask((*A), (*B), (*mask));

  }

 void c_set_with_mask_array_node_node(ArrayPtr*&A){

    assert(A != NULL && *A != NULL && "null pointer found!");
    NodePtr& root = oa_build_tree();
    NodePtr& B = root->input(0);
    NodePtr& mask = root->input(1);
    ArrayPtr ap_b = eval(B);
    ArrayPtr ap_mask = eval(mask);
    
    oa::funcs::set_with_mask((*A), ap_b, ap_mask);
 
 }

}
