
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../tree_tool/NodeVec.hpp"

//sqx modify


void new_node_max_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MAX, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_min_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MIN, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_min_at_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MIN_AT, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_max_at_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MAX_AT, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_abs_max_at_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_ABS_MAX_AT, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_abs_min_at_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_ABS_MIN_AT, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_abs_max_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_ABS_MAX, id1);

	*res = NodeVec::global()->index();
	return;
}

void new_node_abs_min_simple(int *res, int *id1) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_ABS_MIN, id1);

	*res = NodeVec::global()->index();
	return;
}


void new_node_max2_simple(int *res, int*id1, int*id2) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MAX2, id1, id2);

	*res = NodeVec::global()->index();
	return;
}

void new_node_min2_simple(int *res, int*id1, int*id2) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_MIN2, id1, id2);

	*res = NodeVec::global()->index();
	return;
}


extern "C" {

	void c_new_node_max_simple(int *res, int *id1)
	{
		new_node_max_simple(res, id1);
	}

	void c_new_node_min_simple(int *res, int *id1)
	{
		new_node_min_simple(res, id1);
	}

	void c_new_node_min_at_simple(int *res, int *id1)
	{
		new_node_min_at_simple(res, id1);
	}

	void c_new_node_max_at_simple(int *res, int *id1)
	{
		new_node_max_at_simple(res, id1);
	}

	void c_new_node_abs_max_at_simple(int *res, int *id1)
	{
		new_node_abs_max_at_simple(res, id1);
	}

	void c_new_node_abs_min_at_simple(int *res, int *id1)
	{
		new_node_abs_min_at_simple(res, id1);
	}

	void c_new_node_abs_max_simple(int *res, int *id1)
	{
		new_node_abs_max_simple(res, id1);
	}

	void c_new_node_abs_min_simple(int *res, int *id1)
	{
		new_node_abs_min_simple(res, id1);
	}


	void c_new_node_max2_simple(int* res, int *id1, int *id2)
	{
		new_node_max2_simple(res, id1, id2);
	}

	void c_new_node_min2_simple(int* res, int *id1, int *id2)
	{
		new_node_min2_simple(res, id1, id2);
	}


}



