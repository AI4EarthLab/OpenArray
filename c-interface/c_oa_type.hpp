#ifndef __C_OA_TYPE_HPP__
#define __C_OA_TYPE_HPP__

#include "../ArrayPool.hpp"
#include "../NodePool.hpp"
#include "../Grid.hpp"

#define L 0
#define R 1

extern "C" {

  void c_array_assign_array(ArrayPtr*& A, ArrayPtr*& B,
          int & pa, int & pb);  

  void c_node_assign_node(NodePtr* &A, NodePtr* &B);

  void c_node_assign_array(ArrayPtr* &A, NodePtr* &B);

  void c_destroy_array(void*& A);

  void c_destroy_node(void*& A);

  void c_display_array_info(ArrayPtr* A, char* prefix);

  void c_display_array(ArrayPtr* A, char* prefix);

  void gdb_display_range(ArrayPtr* A, int is, int ie, int js, int je, int ks, int ke);

  void gdb_display(ArrayPtr* A);

  void c_display_node(NodePtr* A, char* prefix);

  ///:for f in ['ones', 'zeros', 'rands', 'seqs']
  void c_${f}$(ArrayPtr* & ptr, int m, int n, int k,
          int stencil_width = 1, 
          int data_type = DATA_DOUBLE);
  ///:endfor
  
  ///:mute
  ///:set TYPE = [['int'], ['float'], ['double']]
  ///:endmute
  
  ///:for t in TYPE
  void c_consts_${t[0]}$(ArrayPtr* &ptr,
          int m, int n, int k, ${t[0]}$ val, 
          int stencil_width);
  ///:endfor

  ///:for t in TYPE
  void c_new_seqs_scalar_node_${t[0]}$(NodePtr* &ptr,
          ${t[0]}$ val);  
  ///:endfor

  void c_new_node_array(NodePtr* &ptr, ArrayPtr* &ap);

  void c_new_node_op2(NodePtr* &ptr, int nodetype,
          NodePtr* &u, NodePtr* &v);

  void c_new_local_int3(NodePtr* &B, int* val);

  void c_grid_init (char* ch, const ArrayPtr*& A,
          const ArrayPtr*& B, const ArrayPtr*& C);
  
  void c_grid_bind(ArrayPtr*& A, int pos);

  ///:for t in TYPE
  void c_local_sub_${t[0]}$(ArrayPtr*& A, int* x, int* y, int* z, ${t[0]}$* ans); 
  ///:endfor

  ///:for t in TYPE
  void c_set_local_${t[0]}$(ArrayPtr*& A, int* x, int* y, int* z, ${t[0]}$* val);
  ///:endfor

  void c_get_box_corners(ArrayPtr*& A, int* s);

  void c_update_ghost(ArrayPtr*& A);

}

#endif
