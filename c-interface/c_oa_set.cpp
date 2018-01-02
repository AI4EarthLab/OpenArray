#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Internal.hpp"

extern "C"{

  ///:for type1 in ['int','float','double']
  void c_set_ref_const_${type1}$(NodePtr*& A, ${type1}$ val){
    assert(A != NULL && *A != NULL && "null pointer found!");
    oa::funcs::set_ref_const((*A)->input(0)->get_data(),
            (*A)->get_ref(), val);
  }

  //set a const value to an array.
  void c_set_array_const_${type1}$(ArrayPtr*& A, ${type1}$ val){
    assert(A != NULL && *A != NULL && "null pointer found!");
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


  void c_set_ref_array(NodePtr*& A, ArrayPtr*& B){
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(B != NULL && *B != NULL && "null pointer found!");
    ArrayPtr& p = (*A)->input(0)->get_data();
    oa::funcs::set(p, (*A)->get_ref(), *B);
  }

  void c_set_ref_ref(NodePtr*& A, NodePtr*& B){
    assert(A != NULL && *A != NULL && "null pointer found!");
    assert(B != NULL && *B != NULL && "null pointer found!");
    
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

    //oa::utils::print_data(arr, sh,
    //   oa::utils::to_type<${type}$>());

    oa::funcs::set<${type}$>(ap, ref_box, arr, sh);
  }
  ///:endfor

  ///:for type in ['int', 'float', 'double']
  void c_set_${type}$_array(${type}$& val,ArrayPtr*& ap){
    ENSURE_VALID_PTR(ap);
    if((*ap)->shape() != SCALAR_SHAPE){
      std::cout<<(boost::format("can not covert input "
              "array of shape (%1%,%2%,%3%) to a scalar.")
              % (*ap)->shape()[0]
              % (*ap)->shape()[1]
              % (*ap)->shape()[2]).str()
               << std::endl;
      exit(0);
    }

    ArrayPtr ap1;
    
    if(!(*ap)->is_seqs()){
      ap1 = oa::funcs::g2l(*ap);    
    }else{
      ap1 = *ap;
    }

    switch(ap1->get_data_type()){
    case DATA_INT:
      val = ((int*)ap1->get_buffer())[0];
      break;
    case DATA_FLOAT:
      val = ((float*)ap1->get_buffer())[0];
      break;
    case DATA_DOUBLE:
      val = ((double*)ap1->get_buffer())[0];      
      break;
    }
  }
  ///:endfor

  ///:for type in ['int', 'float', 'double']
  void c_set_${type}$_node(${type}$& val, NodePtr*& np){
    ENSURE_VALID_PTR(np);
    if((*np)->shape() != SCALAR_SHAPE){
      std::cout<<(boost::format("can not covert input "
                      "node of shape (%1%,%2%,%3%) to a scalar.")
              % (*np)->shape()[0]
              % (*np)->shape()[1]
              % (*np)->shape()[2]).str()
               << std::endl;
      exit(0);
    }
    
    try{
      ArrayPtr ap = oa::ops::eval(*np);
      ArrayPtr ap1 = oa::funcs::g2l(ap);
    
      switch(ap1->get_data_type()){
      case DATA_INT:
        val = ((int*)ap1->get_buffer())[0];
        break;
      case DATA_FLOAT:
        val = ((float*)ap1->get_buffer())[0];
        break;
      case DATA_DOUBLE:
        val = ((double*)ap1->get_buffer())[0];      
        break;
      }
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }
  ///:endfor

}
