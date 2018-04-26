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
    
    if ((*B)->is_ref()) {
      ArrayPtr& p1 = (*A)->input(0)->get_data();
      ArrayPtr& p2 = (*B)->input(0)->get_data();
    
      oa::funcs::set(p1, (*A)->get_ref(), p2, (*B)->get_ref());
    } else {
      ArrayPtr& p1 = (*A)->input(0)->get_data();
      oa::ops::gen_kernels_JIT_with_op(*B);
      ArrayPtr p2 = oa::ops::eval(*B);

      oa::funcs::set(p1, (*A)->get_ref(), p2);
    }
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

    // std::cout<<"##"<<s[0]<<"  "<<s[1]<<" "<<s[2]<<std::endl;
    // for(int i = 0; i < s[0] * s[1] * s[2]; ++i){
    //   std::cout<<i<<" : "<<arr[i]<<std::endl;
    // }
    // ref_box.display("ref_box");
    
    oa::funcs::set<${type}$>(ap, ref_box, arr, sh);

  }
  ///:endfor

  ///:for type in ['int', 'float', 'double']
  ///:for src_type in [['node', 'NodePtr'], ['array', 'ArrayPtr']]
  void c_set_farray_${src_type[0]}$_${type}$(${type}$*& arr,
          ${src_type[1]}$*& p, int* s){
    
    Shape arr_sh;
    arr_sh[0] = s[0]; arr_sh[1] = s[1]; arr_sh[2] = s[2];

    if(arr_sh != (*p)->shape()){
      printf("src shape(%d,%d,%d) and dst shape(%d,%d,%d) does not match\n",
                  (*p)->shape()[0],
                  (*p)->shape()[1],
                  (*p)->shape()[2],
                  arr_sh[0],
                  arr_sh[1],
                  arr_sh[2]);
      exit(0);
    }
    
    const Box arr_box(0, arr_sh[0], 0, arr_sh[1], 0, arr_sh[2]);
    
    ArrayPtr ap1, ap;

    ///:if src_type[0] == 'array'
    ap = *p;
    ///:else
    try{
      ap = oa::ops::eval(*p);
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
    ///:endif

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    void * buf = ap1->get_buffer();
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();
    
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<${type}$, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<${type}$, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<${type}$, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
  ///:endfor
  ///:endfor
  
  ///:for type in ['int', 'float', 'double']
  void c_set_${type}$_array(${type}$& val,ArrayPtr*& ap){
    ENSURE_VALID_PTR(ap);
    if((*ap)->shape() != SCALAR_SHAPE){
      printf("can not covert input array of shape (%d,%d,%d) to a scalar.\n",
              (*ap)->shape()[0],
              (*ap)->shape()[1],
              (*ap)->shape()[2]);
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
      printf("can not covert intpu node of shape (%d,%d,%d) to a scalar.\n",
              (*np)->shape()[0],
              (*np)->shape()[1],
              (*np)->shape()[2]);
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
