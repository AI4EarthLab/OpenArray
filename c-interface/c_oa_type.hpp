#ifndef __C_OA_TYPE_HPP__
#define __C_OA_TYPE_HPP__

#include "../ArrayPool.hpp"
#include "../NodePool.hpp"

#define L 0
#define R 1

extern "C" {

  void c_array_assign_array(void* &A, void* &B, int & pa, int & pb);

  void c_node_assign_node(void* &A, void* &B);

  void c_node_assign_array(void* &A, void* &B);

  void c_destroy_array(void*& A);

  void c_destroy_node(void*& A);

  void c_display_array(void* A, void* prefix);

  void c_display_node(void* A, void* prefix);

  void c_ones(void* & ptr, int m, int n, int k, int stencil_width = 1, 
    int data_type = DATA_DOUBLE, 
    MPI_Fint fcomm = 0);

  void c_zeros(void* & ptr, int m, int n, int k, int stencil_width = 1, 
    int data_type = DATA_DOUBLE, 
    MPI_Fint fcomm = 0);

  void c_rand(void* & ptr, int m, int n, int k, int stencil_width = 1, 
    int data_type = DATA_DOUBLE, 
    MPI_Fint fcomm = 0);

  void c_seqs(void* & ptr, int m, int n, int k, int stencil_width = 1, 
    int data_type = DATA_DOUBLE, 
    MPI_Fint fcomm = 0);
    
  ///:mute
  ///:set TYPE = [['int'], ['float'], ['double']]
  ///:endmute
  ///:for t in TYPE
  void c_consts_${t[0]}$(void* &ptr, int m, int n, int k, ${t[0]}$ val, 
    int stencil_width = 1, MPI_Fint fcomm = 0);

  ///:endfor

  ///:for t in TYPE
  void c_new_seqs_scalar_node_${t[0]}$(void* &ptr, ${t[0]}$ val, 
    MPI_Fint fcomm = 0);

  ///:endfor

  void c_new_node_array(void* &ptr, void* &ap);

  void c_new_node_op2(void* &ptr, int nodetype, void* &u, void* &v);

  void c_new_node_op1(void* &ptr, int nodetype, void* &u);
}

#endif
