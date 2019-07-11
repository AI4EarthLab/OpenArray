#ifndef __C_OA_TYPE_HPP__
#define __C_OA_TYPE_HPP__

#include "../ArrayPool.hpp"
#include "../NodePool.hpp"
#include "../Grid.hpp"
// xiaogang
#include "c_simple_type.hpp"
#define L 0
#define R 1

extern "C" {

  void c_array_assign_array(ArrayPtr*& A, ArrayPtr*& B,
          int & pa, int & pb);  

  void c_node_assign_node(NodePtr* &A, NodePtr* &B);

  void c_node_assign_array(ArrayPtr* &A);

  void c_destroy_array(void*& A);

  void c_destroy_node(void*& A);

  void c_display_array_info(ArrayPtr* A, char* prefix);

  void c_display_array(ArrayPtr* A, char* prefix);

  void gdb_display_range(ArrayPtr* A, int is, int ie, int js, int je, int ks, int ke);

  void gdb_display(ArrayPtr* A);

  void c_display_node(NodePtr* A, char* prefix);

  void c_ones(ArrayPtr* & ptr, int m, int n, int k,
          int stencil_width = 1, 
          int data_type = DATA_DOUBLE);
  void c_zeros(ArrayPtr* & ptr, int m, int n, int k,
          int stencil_width = 1, 
          int data_type = DATA_DOUBLE);
  void c_rands(ArrayPtr* & ptr, int m, int n, int k,
          int stencil_width = 1, 
          int data_type = DATA_DOUBLE);
  void c_seqs(ArrayPtr* & ptr, int m, int n, int k,
          int stencil_width = 1, 
          int data_type = DATA_DOUBLE);
  
  
  void c_consts_int(ArrayPtr* &ptr,
          int m, int n, int k, int val, 
          int stencil_width);
  void c_consts_float(ArrayPtr* &ptr,
          int m, int n, int k, float val, 
          int stencil_width);
  void c_consts_double(ArrayPtr* &ptr,
          int m, int n, int k, double val, 
          int stencil_width);

  void c_new_seqs_scalar_node_int(NodePtr* &ptr,
          int val);  
  void c_new_seqs_scalar_node_float(NodePtr* &ptr,
          float val);  
  void c_new_seqs_scalar_node_double(NodePtr* &ptr,
          double val);  

  void c_new_node_array(NodePtr* &ptr, ArrayPtr* &ap);

  void c_new_node_op2(NodePtr* &ptr, int nodetype,
          NodePtr* &u, NodePtr* &v);

  void c_new_local_int3(NodePtr* &B, int* val);

  void c_grid_init (char* ch, const ArrayPtr*& A,
          const ArrayPtr*& B, const ArrayPtr*& C);
  
  void c_grid_bind(ArrayPtr*& A, int pos);

  void c_local_sub(ArrayPtr*& A, int* x, int* y, int* z, double* ans); 
/*
  void c_local_sub_int(ArrayPtr*& A, int* x, int* y, int* z, int* ans); 
  void c_local_sub_float(ArrayPtr*& A, int* x, int* y, int* z, float* ans); 
  void c_local_sub_double(ArrayPtr*& A, int* x, int* y, int* z, double* ans); 
*/
  void c_set_local_int(ArrayPtr*& A, int* x, int* y, int* z, int* val);
  void c_set_local_float(ArrayPtr*& A, int* x, int* y, int* z, float* val);
  void c_set_local_double(ArrayPtr*& A, int* x, int* y, int* z, double* val);

  void c_get_box_corners(ArrayPtr*& A, int* s);

  void c_update_ghost(ArrayPtr*& A);

}

#endif
