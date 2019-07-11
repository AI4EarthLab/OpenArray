#ifndef __C_SIMPLE_TYPE_HPP__
#define __C_SIMPLE_TYPE_HPP__

#include "../ArrayPool.hpp"
#include "../common.hpp"
#include "../NodePool.hpp"
#include "../Grid.hpp"
#include "../modules/tree_tool/NodeVec.hpp"

#define L 0
#define R 1

extern "C" {
	void c_new_node_array_simple(ArrayPtr* &ap, int *res);

	
  void c_new_seqs_scalar_node_int_simple(int val, int *res);
	
  void c_new_seqs_scalar_node_float_simple(float val, int *res);
	
  void c_new_seqs_scalar_node_double_simple(double val, int *res);
	
	
  void c_new_node_int3_simple_rep(int* x, int* y, int* z, int *res);
	
  void c_new_node_int3_simple_shift(int* x, int* y, int* z, int *res);
  void c_new_node_op2_simple(int nodetype, int *res_id, int *id1, int *id2);
}

#endif
