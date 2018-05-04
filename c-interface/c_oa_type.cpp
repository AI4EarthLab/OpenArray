#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Kernel.hpp"
#include "../op_define.hpp"
#include "../MPI.hpp"

extern "C" {

  void c_array_assign_array(ArrayPtr*& A, ArrayPtr*& B,
          int & pa, int & pb) {
    if(A == NULL) A = new ArrayPtr();
    
    if(B == NULL) {
      A = NULL;
      return;
    }
    
    //(*(ArrayPtr*&)B)->display("B = ");
    
    if (pb == R) {
      *A = *B;

#ifdef DEBUG            
      ArrayPool::global()->show_status("x1");
      printf("B's hash = %d\n", (*B)->get_hash());
#endif

      //drop object B
      *B = nullptr;
      
#ifdef DEBUG      
      ArrayPool::global()->show_status("x2");
#endif
      
    } else {
      //need to rewrite!
      //Array::copy(*A, *B);

      int dt = (*(ArrayPtr*) B)->get_data_type();
      
      *A = ArrayPool::global()->get(
          (*(ArrayPtr*) B)->get_partition(), dt);
      
      // should copy pos from B
      (*(ArrayPtr*) A)->set_pos( (*(ArrayPtr*)B)->get_pos() );

      switch(dt) {
      case DATA_INT:
        oa::internal::copy_buffer(
            (int*) ((*A)->get_buffer()),
            (int*) ((*(ArrayPtr*) B)->get_buffer()),
            (*A)->buffer_size());
        break;
      case DATA_FLOAT:
        oa::internal::copy_buffer(
            (float*) ((*A)->get_buffer()),
            (float*) ((*(ArrayPtr*) B)->get_buffer()),
            (*A)->buffer_size());
        break;
      case DATA_DOUBLE:
        oa::internal::copy_buffer(
            (double*) ((*A)->get_buffer()),
            (double*) ((*(ArrayPtr*) B)->get_buffer()),
            (*A)->buffer_size());
        break;
      }
    }
    pa = L;
  }

  void c_node_assign_node(NodePtr*&A, NodePtr*&B) {
    if(A == NULL) A = new NodePtr();
    *A = *B;
  }

  void c_node_assign_array(ArrayPtr* &A, NodePtr* &B){
    if(B == NULL || *B == NULL) return;
    if(A == NULL) A = new ArrayPtr();
    
    try{
      // cout<<g_cache<<endl;
      if(g_debug) {
	oa::ops::write_graph(*B, true, "B.dot");
      }

      if (!g_cache) oa::ops::gen_kernels_JIT_with_op(*(NodePtr*)B);
      *A = oa::ops::eval(*(NodePtr*)B);
      g_cache = false;
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }
  
  void c_destroy_array(void*& A) {
    // cout<<"destroy_array called"<<endl;
    // ArrayPool::global()->show_status("y1");
    
    try{
      if (A != NULL) {
        delete((ArrayPtr*) A);
        A = NULL;
      }      
    }catch(const std::exception& e){
      std::cout<<"Exception occured while destroying array. "
        "Message: "<<e.what()<<std::endl;
    }
    // ArrayPool::global()->show_status("y2");     
  }

  void c_destroy_node(void*& A) {
    try{
      if (A != NULL) {
        delete((NodePtr*)A);
        A = NULL;
      }      
    }catch(const std::exception& e){
      std::cout<<"Exception occured while destroying node. "
        "Message: "<<e.what()<<std::endl;
    }
  }

  void c_display_array_info(ArrayPtr* A, char* prefix) {
    //printf("prefix = %s\n", (char*)prefix);
    if(A != NULL){
      (*(ArrayPtr*) A)->display_info((char*)prefix);      
    }
  }

  void c_display_array(ArrayPtr* A, char* prefix) {
    //printf("prefix = %s\n", (char*)prefix);
    if(A != NULL){
      (*(ArrayPtr*) A)->display((char*)prefix);      
    }
  }

  void gdb_display_range(ArrayPtr* A, int is, int ie, int js, int je, int ks, int ke) {
    int a = is;
    int b = ie;
    int c = js;
    int d = je;
    int e = ks;
    int f = ke;
    if(A != NULL){
      (*(ArrayPtr*) A)->display(NULL,a,b,c,d,e,f);      
    }
  }

  void gdb_display(ArrayPtr* A) {
    if(A != NULL){
      (*(ArrayPtr*) A)->display(NULL);      
    }
  }

  void c_display_node(NodePtr* A, char* prefix) {
    (*(NodePtr*) A)->display((char*) prefix);
  }


  ///:for f in ['ones', 'zeros', 'rands', 'seqs']
  void c_${f}$(ArrayPtr* & ptr,
          int m, int n, int k, int stencil_width, 
          int data_type) {

    if(ptr == NULL) ptr = new ArrayPtr();  
    MPI_Comm comm = oa::MPI::global()->comm();
    Shape s = {m, n, k};
    *ptr = oa::funcs::${f}$(comm, s, stencil_width, data_type);
  }
  ///:endfor


  ///:mute
  ///:set TYPE = [['int'], ['float'], ['double']]
  ///:endmute

  ///:for t in TYPE
  void c_consts_${t[0]}$(ArrayPtr* &ptr,
          int m, int n, int k, ${t[0]}$ val, 
          int stencil_width) {
    MPI_Comm comm = oa::MPI::global()->comm();
    Shape s = {m, n, k};

    if (ptr == NULL) ptr = new ArrayPtr();
    *ptr = oa::funcs::consts(comm, s, val, stencil_width);
  }
  ///:endfor

  ///:for t in TYPE
  void c_new_seqs_scalar_node_${t[0]}$(NodePtr* &ptr,
          ${t[0]}$ val) {

    if (ptr == NULL) ptr = new NodePtr();
    *ptr = oa::ops::new_seqs_scalar_node(val);
  }
  ///:endfor

  void c_new_node_array(NodePtr* &ptr, ArrayPtr* &ap) {
    assert(ap != NULL &&
            "array pointer can not be null to create a node.");

    if (ptr == NULL) ptr = new NodePtr();
    
    *ptr = oa::ops::new_node(*(ArrayPtr*)ap);
  }

  void c_new_node_op2(NodePtr* &ptr, int nodetype,
          NodePtr* &u, NodePtr* &v) {
    if (ptr == NULL) ptr = new NodePtr();
    
    *ptr = oa::ops::new_node((NodeType)nodetype, *u, *v);
  }

  void c_new_local_int3(NodePtr* &ptr, int* val){
    if (ptr == NULL) ptr = new NodePtr();

    *ptr = NodePool::global()->get_local_1d<int, 3>(val);
  }


  void c_shape_node(NodePtr*& A, int* s){
    ENSURE_VALID_PTR(A);  
    s[0] = (*A)->shape()[0];
    s[1] = (*A)->shape()[1];
    s[2] = (*A)->shape()[2];
  }

  void c_shape_array(ArrayPtr*& A, int* s){
    ENSURE_VALID_PTR(A);    
    s[0] = (*A)->shape()[0];
    s[1] = (*A)->shape()[1];
    s[2] = (*A)->shape()[2];
  }

  void c_local_shape(ArrayPtr*& A, int* s){
    ENSURE_VALID_PTR(A);    
    s[0] = (*A)->local_shape()[0];
    s[1] = (*A)->local_shape()[1];
    s[2] = (*A)->local_shape()[2];
  }

  void c_buffer_shape(ArrayPtr*& A, int* s){
    ENSURE_VALID_PTR(A);    
    s[0] = (*A)->buffer_shape()[0];
    s[1] = (*A)->buffer_shape()[1];
    s[2] = (*A)->buffer_shape()[2];
  }

  void c_get_buffer_ptr(ArrayPtr*& A, void*& p){
    ENSURE_VALID_PTR(A);    
    p = (*A)->get_buffer();
  }

  //set stencil width and stencil type
  void c_set_stencil(int st, int sw){
    Partition::set_default_stencil_width(sw);
    Partition::set_default_stencil_type(st);
  }

  void c_get_stencil_width(int& sw){
    sw = Partition::get_default_stencil_width();
  }

  void c_get_stencil_type(int& st){
    st = Partition::get_default_stencil_width();
  }

  ///:for t in ['int', 'float', 'double']
  void c_is_array_${t}$(int& flag, ArrayPtr*& A){
    ENSURE_VALID_PTR(A);
    if((*A)->get_data_type() == DATA_${t.upper()}$){
      flag = 1;
    }else {
      flag = 0;
    }
  }
  ///:endfor

  void c_make_psudo3d(ArrayPtr*& A, ArrayPtr*& B){
    if(A == NULL) A = new ArrayPtr();
    ENSURE_VALID_PTR(B);
    *A = oa::funcs::make_psudo3d(*B);
    // (*A)->display("AAA=");
  }

  void c_format_short(int flag){
    oa::utils::set_disp_format(flag);
  }

  void c_print_c_ptr(ArrayPtr*& d){
    printf("pointer is %p\n", d->get());
  }

  void c_get_box_corners(ArrayPtr*& A, int* s){
    ENSURE_VALID_PTR(A);
    (*A)->get_local_box().get_corners(
        s[0], s[1], s[2], s[3], s[4], s[5]
    );
  }
  
  ///:for t in TYPE 
  void c_local_sub_${t[0]}$(ArrayPtr*& A, int* x, int* y, int* z, ${t[0]}$* ans) {
    oa::funcs::local_sub(*A, *x, *y, *z, ans);
  }

  ///:endfor
  
  ///:for t in TYPE
  void c_set_local_${t[0]}$(ArrayPtr*& A, int* x, int* y, int* z, ${t[0]}$* val) {
    oa::funcs::set_local(*A, *x, *y, *z, *val);
  }

  ///:endfor
  
  void c_update_ghost(ArrayPtr*& A) {
    oa::funcs::update_ghost(*A);
  }

}

