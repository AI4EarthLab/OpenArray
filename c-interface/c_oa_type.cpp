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
    
    // printf("A=%p\n", A);
    // printf("B=%p\n", B);
    
    if(B == NULL) return;

    //(*(ArrayPtr*&)B)->display("B = ");
    
    if (pb == R) {
      *A = *B;
      *B = NULL;
    } else {
      int dt = (*(ArrayPtr*) B)->get_data_type();
      *A = ArrayPool::global()->get(
          (*(ArrayPtr*) B)->get_partition(), dt);

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
            (*A)->buffer_size()
                                  );
        break;
      case DATA_DOUBLE:
        oa::internal::copy_buffer(
            (double*) ((*A)->get_buffer()),
            (double*) ((*(ArrayPtr*) B)->get_buffer()),
            (*A)->buffer_size());
        break;
      }
      // ArrayPtr* tmp = new ArrayPtr();
      // *tmp = ap;
      // A = (void*) tmp;
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
      *A = oa::ops::eval(*(NodePtr*)B);
    }catch(const std::exception& e){
      std::cout<<"Execetion caught while "
        "executing eval function. "
        "Message: "<<e.what()<<std::endl;
      exit(0);
    }
  }
  
  void c_destroy_array(void*& A) {
    //cout<<"destroy_array called"<<endl;
    try{
      if (A != NULL) {
        delete((ArrayPtr*) A);
        A = NULL;
      }      
    }catch(const std::exception& e){
      std::cout<<"Exception occured whilg destroying array. "
        "Message: "<<e.what()<<std::endl;
    }
  }

  void c_destroy_node(void*& A) {
    try{
      if (A != NULL) {
        delete((NodePtr*)A);
        A = NULL;
      }      
    }catch(const std::exception& e){
      std::cout<<"Exception occured whilg destroying node. "
        "Message: "<<e.what()<<std::endl;
    }
  }

  void c_display_array(ArrayPtr* A, char* prefix) {
    //printf("prefix = %s\n", (char*)prefix);
    if(A != NULL){
      (*(ArrayPtr*) A)->display((char*)prefix);      
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
  ///:set TYPE = [['int'], &
                ['float'], &
                ['double']]
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

  void c_new_node_op1(NodePtr* &ptr, int nodetype, NodePtr* &u) {
    if (ptr == NULL) ptr = new NodePtr();
    
    *ptr = oa::ops::new_node((NodeType)nodetype, *u);
  }

void c_new_local_int3(NodePtr* &ptr, int* val){
  if (ptr == NULL) ptr = new NodePtr();

  *ptr = NodePool::global()->get_local_1d<int, 3>(val);
}


void c_shape_node(NodePtr*& A, int* s){
  s[0] = (*A)->shape()[0];
  s[1] = (*A)->shape()[1];
  s[2] = (*A)->shape()[2];
}

void c_shape_array(ArrayPtr*& A, int* s){
  s[0] = (*A)->shape()[0];
  s[1] = (*A)->shape()[1];
  s[2] = (*A)->shape()[2];
}


}

