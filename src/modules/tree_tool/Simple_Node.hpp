#include "../../Array.hpp"
#include "../../Node.hpp"

#ifndef _SIMPLE_NODE_HPP__
#define _SIMPLE_NODE_HPP__

union uni_data{
	int union_int3[3];//used for rep shift
	int union_int6[6];//used for sub
	int union_int_slice;//used for slice
	int union_value_int;//used to store int
	float union_value_float;//used to store float
	double union_value_double;//used to store double
};
typedef union uni_data extra_info;

struct Simple_node{

	int type; //TYPE_INT TYPE_FLOAT TYPE_DOUBLE TYPE_SUB TYPE_BOX TYPE_INT3 TYPE_SLICE_INT
	ArrayPtr* array_ptr;
	extra_info info;
	NodePtr ndp;
	int input[2];
	Simple_node()
	{
		type=-1;
		array_ptr = NULL;
		input[0] = -1;
		input[1] = -1;
	}
	~Simple_node(){}


	void* get_val();
	int get_type();
	ArrayPtr get_ArrayPtr();

	void set_ArrayPtr(ArrayPtr* &a);
	void set_ArrayPtr_NULL();

	bool set_info(extra_info info);
	bool set_type(int type);		
};
#endif
