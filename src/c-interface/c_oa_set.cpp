#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Internal.hpp"
#include "../common.hpp"
#include "../TreeRootDict.hpp"
#include "../modules/tree_tool/NodeVec.hpp"

#include "../MPI.hpp"

extern "C"{

  void c_set_ref_const_int( int val){
    //xiaogang
    NodePtr& c_b = oa_build_tree();

		oa::funcs::set_ref_const(c_b->input(0)->get_data(),
            c_b->get_ref(), val);
  }

  //set a const value to an array.
  void c_set_array_const_int(ArrayPtr*& A, int val){
    assert(A != NULL && *A != NULL && "null pointer found!");
    Shape s = (*A)->buffer_shape();
    int size = s[0] * s[1] * s[2];
    DataType dt = (*A)->get_data_type();

    void* buf = (*A)->get_buffer();
#ifndef __HAVE_CUDA__
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
#else
    switch(dt){
    case DATA_INT: 
      oa::gpu::set_buffer_consts((int*)buf,
              size, (int)val);
      break;
    case DATA_FLOAT:
      oa::gpu::set_buffer_consts((float*)buf,
              size, (float)val);
      break;
    case DATA_DOUBLE:
      oa::gpu::set_buffer_consts((double*)buf,
              size, (double)val);
      break;
    }
#endif
  }
  
  void c_set_ref_const_float( float val){
    //xiaogang
    NodePtr& c_b = oa_build_tree();

		oa::funcs::set_ref_const(c_b->input(0)->get_data(),
            c_b->get_ref(), val);
  }

  //set a const value to an array.
  void c_set_array_const_float(ArrayPtr*& A, float val){
    assert(A != NULL && *A != NULL && "null pointer found!");
    Shape s = (*A)->buffer_shape();
    int size = s[0] * s[1] * s[2];
    DataType dt = (*A)->get_data_type();

    void* buf = (*A)->get_buffer();
#ifndef __HAVE_CUDA__
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
#else
    switch(dt){
    case DATA_INT: 
      oa::gpu::set_buffer_consts((int*)buf,
              size, (int)val);
      break;
    case DATA_FLOAT:
      oa::gpu::set_buffer_consts((float*)buf,
              size, (float)val);
      break;
    case DATA_DOUBLE:
      oa::gpu::set_buffer_consts((double*)buf,
              size, (double)val);
      break;
    }
#endif
  }
  
  void c_set_ref_const_double( double val){
    //xiaogang
    NodePtr& c_b = oa_build_tree();

		oa::funcs::set_ref_const(c_b->input(0)->get_data(),
            c_b->get_ref(), val);
  }

  //set a const value to an array.
  void c_set_array_const_double(ArrayPtr*& A, double val){
    assert(A != NULL && *A != NULL && "null pointer found!");
    Shape s = (*A)->buffer_shape();
    int size = s[0] * s[1] * s[2];
    DataType dt = (*A)->get_data_type();

    void* buf = (*A)->get_buffer();
#ifndef __HAVE_CUDA__
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
#else
    switch(dt){
    case DATA_INT: 
      oa::gpu::set_buffer_consts((int*)buf,
              size, (int)val);
      break;
    case DATA_FLOAT:
      oa::gpu::set_buffer_consts((float*)buf,
              size, (float)val);
      break;
    case DATA_DOUBLE:
      oa::gpu::set_buffer_consts((double*)buf,
              size, (double)val);
      break;
    }
#endif
  }
  


  void c_set_ref_array(ArrayPtr*& B){
    assert(B != NULL && *B != NULL && "null pointer found!");
    NodePtr& A = oa_build_tree();
    ArrayPtr& p = A->input(0)->get_data();
    oa::funcs::set(p, A->get_ref(), *B);
  }
	
void c_new_type_set(int *id1, int *id2)
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_SET, id1, id2);
}

  void c_set_ref_ref(){
    //xiaogang
		//assert(A != NULL && *A != NULL && "null pointer found!");
    //assert(B != NULL && *B != NULL && "null pointer found!");
     NodePtr& root =  oa_build_tree();
     NodePtr& A = root->input(0);
     NodePtr& B = root->input(1);

    if (B->is_ref()) {
      ArrayPtr& p1 = A->input(0)->get_data();
      ArrayPtr& p2 = B->input(0)->get_data();
    
      oa::funcs::set(p1, A->get_ref(), p2, B->get_ref());
    } else {
      ArrayPtr& p1 = A->input(0)->get_data();
      //oa::ops::gen_kernels_JIT_with_op(B);
      ArrayPtr p2 = oa::ops::eval(B);

      oa::funcs::set(p1, A->get_ref(), p2);
    }
  }


  void c_set_array_farray_int(
      ArrayPtr*& ap, int*& arr, int* s){
    if(ap == NULL) ap = new ArrayPtr();
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];


    ArrayPtr local =
      oa::funcs::create_local_array<int>(sh, arr);

    *ap = oa::funcs::l2g(local);
  }
  void c_set_array_farray_float(
      ArrayPtr*& ap, float*& arr, int* s){
    if(ap == NULL) ap = new ArrayPtr();
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];


    ArrayPtr local =
      oa::funcs::create_local_array<float>(sh, arr);

    *ap = oa::funcs::l2g(local);
  }
  void c_set_array_farray_double(
      ArrayPtr*& ap, double*& arr, int* s){
    if(ap == NULL) ap = new ArrayPtr();
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];


    ArrayPtr local =
      oa::funcs::create_local_array<double>(sh, arr);

    *ap = oa::funcs::l2g(local);
  }

  void c_set_ref_farray_int( int*& arr, int* s){
    NodePtr& np = oa_build_tree();
    assert(np->is_ref_data());

    ArrayPtr& ap = np->get_ref_data();
    Box& ref_box = np->get_ref();

    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];

    //oa::utils::print_data(arr, sh,
    //   oa::utils::to_type<int>());

    // std::cout<<"##"<<s[0]<<"  "<<s[1]<<" "<<s[2]<<std::endl;
    // for(int i = 0; i < s[0] * s[1] * s[2]; ++i){
    //   std::cout<<i<<" : "<<arr[i]<<std::endl;
    // }
    // ref_box.display("ref_box");

    oa::funcs::set<int>(ap, ref_box, arr, sh);

  }
  void c_set_ref_farray_float( float*& arr, int* s){
    NodePtr& np = oa_build_tree();
    assert(np->is_ref_data());

    ArrayPtr& ap = np->get_ref_data();
    Box& ref_box = np->get_ref();

    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];

    //oa::utils::print_data(arr, sh,
    //   oa::utils::to_type<float>());

    // std::cout<<"##"<<s[0]<<"  "<<s[1]<<" "<<s[2]<<std::endl;
    // for(int i = 0; i < s[0] * s[1] * s[2]; ++i){
    //   std::cout<<i<<" : "<<arr[i]<<std::endl;
    // }
    // ref_box.display("ref_box");

    oa::funcs::set<float>(ap, ref_box, arr, sh);

  }
  void c_set_ref_farray_double( double*& arr, int* s){
    NodePtr& np = oa_build_tree();
    assert(np->is_ref_data());

    ArrayPtr& ap = np->get_ref_data();
    Box& ref_box = np->get_ref();

    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];

    //oa::utils::print_data(arr, sh,
    //   oa::utils::to_type<double>());

    // std::cout<<"##"<<s[0]<<"  "<<s[1]<<" "<<s[2]<<std::endl;
    // for(int i = 0; i < s[0] * s[1] * s[2]; ++i){
    //   std::cout<<i<<" : "<<arr[i]<<std::endl;
    // }
    // ref_box.display("ref_box");

    oa::funcs::set<double>(ap, ref_box, arr, sh);

  }


//cyw modify =============================================================

  void c_set_farray_node_int (int*& arr,int* s){
    
    Shape arr_sh;
    arr_sh[0] = s[0]; arr_sh[1] = s[1]; arr_sh[2] = s[2];

    NodePtr& p = oa_build_tree();
    if(arr_sh != p->shape()){
      printf("src shape(%d,%d,%d) and dst shape(%d,%d,%d) does not match\n",
                  p->shape()[0],
                  p->shape()[1],
                  p->shape()[2],
                  arr_sh[0],
                  arr_sh[1],
                  arr_sh[2]);
      exit(0);
    }
    
    const Box arr_box(0, arr_sh[0], 0, arr_sh[1], 0, arr_sh[2]);
    
    ArrayPtr ap1, ap;

    try{
      ap = oa::ops::eval(p);
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<int, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<int, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<int, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
  void c_set_farray_node_float (float*& arr,int* s){
    
    Shape arr_sh;
    arr_sh[0] = s[0]; arr_sh[1] = s[1]; arr_sh[2] = s[2];

    NodePtr& p = oa_build_tree();
    if(arr_sh != p->shape()){
      printf("src shape(%d,%d,%d) and dst shape(%d,%d,%d) does not match\n",
                  p->shape()[0],
                  p->shape()[1],
                  p->shape()[2],
                  arr_sh[0],
                  arr_sh[1],
                  arr_sh[2]);
      exit(0);
    }
    
    const Box arr_box(0, arr_sh[0], 0, arr_sh[1], 0, arr_sh[2]);
    
    ArrayPtr ap1, ap;

    try{
      ap = oa::ops::eval(p);
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<float, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<float, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<float, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
  void c_set_farray_node_double (double*& arr,int* s){
    
    Shape arr_sh;
    arr_sh[0] = s[0]; arr_sh[1] = s[1]; arr_sh[2] = s[2];

    NodePtr& p = oa_build_tree();
    if(arr_sh != p->shape()){
      printf("src shape(%d,%d,%d) and dst shape(%d,%d,%d) does not match\n",
                  p->shape()[0],
                  p->shape()[1],
                  p->shape()[2],
                  arr_sh[0],
                  arr_sh[1],
                  arr_sh[2]);
      exit(0);
    }
    
    const Box arr_box(0, arr_sh[0], 0, arr_sh[1], 0, arr_sh[2]);
    
    ArrayPtr ap1, ap;

    try{
      ap = oa::ops::eval(p);
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<double, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<double, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<double, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
 
  void c_set_farray_array_int(int*& arr,
          ArrayPtr*& p, int* s){
    
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

    ap = *p;

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    (*p)->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<int, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<int, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<int, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
  void c_set_farray_array_float(float*& arr,
          ArrayPtr*& p, int* s){
    
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

    ap = *p;

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    (*p)->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<float, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<float, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<float, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
  void c_set_farray_array_double(double*& arr,
          ArrayPtr*& p, int* s){
    
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

    ap = *p;

    if(ap->is_seqs()){
      ap1 = ap;
    }else{
      ap1 = oa::funcs::g2l(ap);
    }
    
    #ifdef __HAVE_CUDA__
    ap1->memcopy_gpu_to_cpu();
    void * buf = ap1->get_cpu_buffer();
    #else
    void * buf = ap1->get_buffer();
    #endif
    const int sw = ap1->get_partition()->get_stencil_width();
    const Shape& ap_shape = ap1->buffer_shape();
    const Box ap_box(sw, ap_shape[0]-sw,
            sw, ap_shape[1]-sw,
            sw, ap_shape[2]-sw);

    const DataType dt = ap->get_data_type();

    #ifdef __HAVE_CUDA__
    (*p)->memcopy_gpu_to_cpu();
    #endif
    switch(dt){
    case DATA_INT:
      oa::internal::copy_buffer<double, int>(arr,
              arr_sh, arr_box,
              (int*)buf, ap_shape, ap_box);
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer<double, float>(arr,
              arr_sh, arr_box,
              (float*)buf, ap_shape, ap_box);
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer<double, double>(arr,
              arr_sh, arr_box,
              (double*)buf, ap_shape, ap_box);
      break;      
    }
  }
 

//cyw modify end




  void c_set_int_array(int& val,ArrayPtr*& ap){
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

    #ifndef __HAVE_CUDA__
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
    #else
    ap1->memcopy_gpu_to_cpu();
    switch(ap1->get_data_type()){
    case DATA_INT:
      val = ((int*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_FLOAT:
      val = ((float*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_DOUBLE:
      val = ((double*)ap1->get_cpu_buffer())[0];      
      break;
    }
    #endif
  }
  void c_set_float_array(float& val,ArrayPtr*& ap){
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

    #ifndef __HAVE_CUDA__
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
    #else
    ap1->memcopy_gpu_to_cpu();
    switch(ap1->get_data_type()){
    case DATA_INT:
      val = ((int*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_FLOAT:
      val = ((float*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_DOUBLE:
      val = ((double*)ap1->get_cpu_buffer())[0];      
      break;
    }
    #endif
  }
  void c_set_double_array(double& val,ArrayPtr*& ap){
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

    #ifndef __HAVE_CUDA__
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
    #else
    ap1->memcopy_gpu_to_cpu();
    switch(ap1->get_data_type()){
    case DATA_INT:
      val = ((int*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_FLOAT:
      val = ((float*)ap1->get_cpu_buffer())[0];
      break;
    case DATA_DOUBLE:
      val = ((double*)ap1->get_cpu_buffer())[0];      
      break;
    }
    #endif
  }

  void c_set_int_node(int& val){
    //xiaogang
		NodePtr& np = oa_build_tree();

		ENSURE_VALID_PTR(np);
    if(np->shape() != SCALAR_SHAPE){
      printf("can not covert intpu node of shape (%d,%d,%d) to a scalar.\n",
              np->shape()[0],
              np->shape()[1],
              np->shape()[2]);
      exit(0);
    }
    
    try{
      ArrayPtr ap = oa::ops::eval(np);
      ArrayPtr ap1 = oa::funcs::g2l(ap);

      #ifndef __HAVE_CUDA__
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
      #else
      ap1->memcopy_gpu_to_cpu();
      switch(ap1->get_data_type()){
      case DATA_INT:
        val = ((int*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_FLOAT:
        val = ((float*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_DOUBLE:
        val = ((double*)ap1->get_cpu_buffer())[0];      
        break;
      }
      #endif
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }
  void c_set_float_node(float& val){
    //xiaogang
		NodePtr& np = oa_build_tree();

		ENSURE_VALID_PTR(np);
    if(np->shape() != SCALAR_SHAPE){
      printf("can not covert intpu node of shape (%d,%d,%d) to a scalar.\n",
              np->shape()[0],
              np->shape()[1],
              np->shape()[2]);
      exit(0);
    }
    
    try{
      ArrayPtr ap = oa::ops::eval(np);
      ArrayPtr ap1 = oa::funcs::g2l(ap);

      #ifndef __HAVE_CUDA__
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
      #else
      ap1->memcopy_gpu_to_cpu();
      switch(ap1->get_data_type()){
      case DATA_INT:
        val = ((int*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_FLOAT:
        val = ((float*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_DOUBLE:
        val = ((double*)ap1->get_cpu_buffer())[0];      
        break;
      }
      #endif
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }
  void c_set_double_node(double& val){
    //xiaogang
		NodePtr& np = oa_build_tree();

		ENSURE_VALID_PTR(np);
    if(np->shape() != SCALAR_SHAPE){
      printf("can not covert intpu node of shape (%d,%d,%d) to a scalar.\n",
              np->shape()[0],
              np->shape()[1],
              np->shape()[2]);
      exit(0);
    }
    
    try{
      ArrayPtr ap = oa::ops::eval(np);
      ArrayPtr ap1 = oa::funcs::g2l(ap);

      #ifndef __HAVE_CUDA__
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
      #else
      ap1->memcopy_gpu_to_cpu();
      switch(ap1->get_data_type()){
      case DATA_INT:
        val = ((int*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_FLOAT:
        val = ((float*)ap1->get_cpu_buffer())[0];
        break;
      case DATA_DOUBLE:
        val = ((double*)ap1->get_cpu_buffer())[0];      
        break;
      }
      #endif
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }


  void c_transfer_dfarray_to_array_int(
      ArrayPtr*& ap, int*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */
    int sw = Partition::get_default_stencil_width();

    PartitionPtr pp;
    pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    pp->set_default_procs_shape(nps);


    ArrayPtr informal_ap;
    informal_ap = ArrayPool::global()->get(pp, DATA_DOUBLE);

    // give ap values
    int* temp_buffer = (int*)informal_ap->get_buffer();
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)] =
                    arr[k*M*N + j*M + i];
            }
        }
    }
    
    Shape gs = informal_ap->shape();
    ArrayPtr dest_ap = oa::funcs::zeros(comm, gs, sw, DATA_INT);

    if (!informal_ap->get_partition()->equal(dest_ap->get_partition()))
        *ap = oa::funcs::transfer(informal_ap, dest_ap->get_partition());
    else
        *ap = informal_ap;

    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }
  void c_transfer_array_to_dfarray_int(
      ArrayPtr*& ap, int*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */

    const int sw = Partition::get_default_stencil_width();

    PartitionPtr dest_pp;
    dest_pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    dest_pp->set_default_procs_shape(nps);

    ArrayPtr temp_ap;
    temp_ap = ArrayPool::global()->get(dest_pp, DATA_INT);


    if (!temp_ap->get_partition()->equal((*ap)->get_partition()))
        temp_ap = oa::funcs::transfer((*ap), temp_ap->get_partition());
    else
        temp_ap = *ap;

    // give ap values
    int* temp_buffer = (int*)(temp_ap->get_buffer());
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                arr[k*M*N + j*M + i] = 
                    temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)];
                    
            }
        }
    }
    
    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }
  void c_transfer_dfarray_to_array_float(
      ArrayPtr*& ap, float*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */
    int sw = Partition::get_default_stencil_width();

    PartitionPtr pp;
    pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    pp->set_default_procs_shape(nps);


    ArrayPtr informal_ap;
    informal_ap = ArrayPool::global()->get(pp, DATA_DOUBLE);

    // give ap values
    float* temp_buffer = (float*)informal_ap->get_buffer();
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)] =
                    arr[k*M*N + j*M + i];
            }
        }
    }
    
    Shape gs = informal_ap->shape();
    ArrayPtr dest_ap = oa::funcs::zeros(comm, gs, sw, DATA_FLOAT);

    if (!informal_ap->get_partition()->equal(dest_ap->get_partition()))
        *ap = oa::funcs::transfer(informal_ap, dest_ap->get_partition());
    else
        *ap = informal_ap;

    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }
  void c_transfer_array_to_dfarray_float(
      ArrayPtr*& ap, float*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */

    const int sw = Partition::get_default_stencil_width();

    PartitionPtr dest_pp;
    dest_pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    dest_pp->set_default_procs_shape(nps);

    ArrayPtr temp_ap;
    temp_ap = ArrayPool::global()->get(dest_pp, DATA_FLOAT);


    if (!temp_ap->get_partition()->equal((*ap)->get_partition()))
        temp_ap = oa::funcs::transfer((*ap), temp_ap->get_partition());
    else
        temp_ap = *ap;

    // give ap values
    float* temp_buffer = (float*)(temp_ap->get_buffer());
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                arr[k*M*N + j*M + i] = 
                    temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)];
                    
            }
        }
    }
    
    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }
  void c_transfer_dfarray_to_array_double(
      ArrayPtr*& ap, double*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */
    int sw = Partition::get_default_stencil_width();

    PartitionPtr pp;
    pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    pp->set_default_procs_shape(nps);


    ArrayPtr informal_ap;
    informal_ap = ArrayPool::global()->get(pp, DATA_DOUBLE);

    // give ap values
    double* temp_buffer = (double*)informal_ap->get_buffer();
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)] =
                    arr[k*M*N + j*M + i];
            }
        }
    }
    
    Shape gs = informal_ap->shape();
    ArrayPtr dest_ap = oa::funcs::zeros(comm, gs, sw, DATA_DOUBLE);

    if (!informal_ap->get_partition()->equal(dest_ap->get_partition()))
        *ap = oa::funcs::transfer(informal_ap, dest_ap->get_partition());
    else
        *ap = informal_ap;

    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }
  void c_transfer_array_to_dfarray_double(
      ArrayPtr*& ap, double*& arr, int* s, 
      int* np, int* box){
    
    MPI_Comm comm = oa::MPI::global()->comm();
    int my_rank = oa::MPI::global()->rank(comm);
    int mpi_size = oa::MPI::global()->size(comm);

    if(ap == NULL) ap = new ArrayPtr();
    const int np_x = np[0];
    const int np_y = np[1];
    const int np_z = np[2];
    const int xs = box[0];
    const int xe = box[1];
    const int ys = box[2];
    const int ye = box[3];
    const int zs = box[4];
    const int ze = box[5];
    
    Shape sh;
    sh[0] = s[0]; sh[1] = s[1]; sh[2] = s[2];
    const int M = sh[0];
    const int N = sh[1];
    const int P = sh[2];
    const int local_size = sh[0]*sh[1]*sh[2];

    //printf("MPI=%d, shape[%d,%d,%d], (%d,%d,%d,%d,%d,%d)\n", my_rank, sh[0], sh[1], sh[2], xs,xe,ys,ye,zs,ze);
    assert(local_size >= 0 &&
            sh[0]==xe-xs && sh[1]==ye-ys && sh[2]==ze-zs &&
            "shape error in c_transfer_dfarray_to_array_double");

    // collect distributed array info
    int dfa_local_info[6] = {xs, xe, ys, ye, zs, ze};
    int* dfa_global_info = (int*)malloc(sizeof(int)*mpi_size*6);;

    MPI_Gather(dfa_local_info, 6, MPI_INT, 
                dfa_global_info, 6, MPI_INT,
                0, comm);

    MPI_Bcast(dfa_global_info, mpi_size*6, MPI_INT, 0, comm);

    // generate informal partition 
    vector<int> lx, ly, lz;
    for (int i=0; i < mpi_size; i++){
        int i_xs = dfa_global_info[i*6 + 0];
        int i_xe = dfa_global_info[i*6 + 1];
        int i_ys = dfa_global_info[i*6 + 2];
        int i_ye = dfa_global_info[i*6 + 3];
        int i_zs = dfa_global_info[i*6 + 4];
        int i_ze = dfa_global_info[i*6 + 5];
        int i_size = (i_xe-i_xs)*(i_ye-i_ys)*(i_ze-i_zs);
        
        if (i_size == 0) break;
        
        if (i < np_x && lx.size() < np_x)
            lx.push_back(i_xe - i_xs);
        if (i % np_x && ly.size() < np_y)
            ly.push_back(i_ye - i_ys);
        if (i % (np_x*np_y) && lz.size() < np_z)
            lz.push_back(i_ze - i_zs);
    }
    /*
    if (my_rank == 0){
        //cout<< lx <<endl;
        //cout<< ly <<endl;
        //cout<< lz <<endl;
        for (int i = 0; i < lx.size(); i++) printf("lx[%d] = %d\n", i, lx[i]);
        for (int i = 0; i < ly.size(); i++) printf("ly[%d] = %d\n", i, ly[i]);
        for (int i = 0; i < lz.size(); i++) printf("lz[%d] = %d\n", i, lz[i]);
    }
    */

    const int sw = Partition::get_default_stencil_width();

    PartitionPtr dest_pp;
    dest_pp = PartitionPool::global()->get(comm, lx, ly, lz, sw);
    
    Shape nps;
    nps[0] = np_x; nps[1] = np_y; nps[2] = np_z;
    dest_pp->set_default_procs_shape(nps);

    ArrayPtr temp_ap;
    temp_ap = ArrayPool::global()->get(dest_pp, DATA_DOUBLE);


    if (!temp_ap->get_partition()->equal((*ap)->get_partition()))
        temp_ap = oa::funcs::transfer((*ap), temp_ap->get_partition());
    else
        temp_ap = *ap;

    // give ap values
    double* temp_buffer = (double*)(temp_ap->get_buffer());
    for (int k = 0; k < P; k++){
        for (int j = 0; j < N; j++){
            for (int i = 0; i < M; i++){
                arr[k*M*N + j*M + i] = 
                    temp_buffer[(k+sw)*(M+2*sw)*(N+2*sw) + (j+sw)*(M+2*sw) + (i+sw)];
                    
            }
        }
    }
    
    // printf("c ptr = %p\n", *ap);

    free(dfa_global_info);
    return;
  }


}
