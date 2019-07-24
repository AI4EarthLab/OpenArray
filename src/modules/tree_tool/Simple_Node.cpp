#include "Simple_Node.hpp"

void* Simple_node::get_val()
{
	if(type == TYPE_REF)
		return (void*) &(info.union_int6);
	if(type == TYPE_UNKNOWN)
		return (void*) &(info.union_int_slice);

	if(type == TYPE_INT)
		return (void*) &(info.union_value_int);

	if(type == TYPE_FLOAT)
		return (void*) &(info.union_value_float);

	if(type == TYPE_DOUBLE)
		return (void*) &(info.union_value_double);

	if(type == TYPE_INT3_SHIFT || type == TYPE_INT3_REP)
		return (void*) &(info.union_int3);

	if(type == TYPE_SET)
		return (void*) &(info.union_int6);

	//if(type == TYPE_REP)
	//	return (void*) &(info.union_int3);

	//if(type == TYPE_SHIFT)
	//	return (void*) &(info.union_int3);
}

int Simple_node::get_type()
{
	return type;
}

ArrayPtr Simple_node::get_ArrayPtr()
{
	return *array_ptr;
}

void Simple_node::set_ArrayPtr(ArrayPtr* &a)
{
	array_ptr = a;
}

void Simple_node::set_ArrayPtr_NULL()
{
	array_ptr = NULL;
}

bool Simple_node::set_info(extra_info info)
{
	this->info = info;
	return true;
}

bool Simple_node::set_type(int type)
{
	this->type = type;
	return true;
}
