#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Internal.hpp"

extern "C"{

  ///:for type1 in ['int','float','double']
  //set a const value to an array.
  void c_set_with_mask_array_const_${type1}$_array(ArrayPtr*& A,
          ${type1}$ val, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    int sw = (*A)->get_partition()->get_stencil_width();

    ArrayPtr B = oa::funcs::get_seq_scalar(val);

    oa::funcs::set_with_mask((*A), B, (*mask));

  }
  ///:endfor

  void c_set_with_mask_array_array_array(ArrayPtr*& A,
          ArrayPtr*& B, ArrayPtr*& mask){
    
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(B != NULL && *B != NULL && "null pointer found!");
    assert(mask != NULL && *mask != NULL && "null pointer found!");

    oa::funcs::set_with_mask((*A), (*B), (*mask));

  }

}
