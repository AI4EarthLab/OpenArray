
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../tree_tool/NodeVec.hpp"
#include "../tree_tool/Simple_Node.hpp"
//sqx modify

void new_node_sub_simple(int ra0, int ra1, int rb0, int rb1, int rc0, int rc1, int *res, int* id1)
{
	extra_info info;

	info.union_int6[0] = ra0; 	info.union_int6[1] = ra1;
	info.union_int6[2] = rb0; 	info.union_int6[3] = rb1;
	info.union_int6[4] = rc0; 	info.union_int6[5] = rc1;

	NodeVec::global()->set_datanode_size();

	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();
	int datanode_size = NodeVec::global()->get_datanode_size();
	int datavec_size = NodeVec::global()->get_datavec_size();

	if(datanode_size > datavec_size)
		NodeVec::global()->reset_datavec();
	
	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_datavec(TYPE_REF, id1, info);
	NodeVec::global()->set_nodevec(TYPE_REF, id1, info);

	*res = NodeVec::global()->index();
}

void new_node_slice_simple(int k, int* res_id, int* id) 
{
	extra_info info;
	info.union_int_slice = k;

	NodeVec::global()->set_datanode_size();

	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();
	int datanode_size = NodeVec::global()->get_datanode_size();
	int datavec_size = NodeVec::global()->get_datavec_size();

	if(datanode_size > datavec_size)
		NodeVec::global()->reset_datavec();
	
	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_datavec(TYPE_UNKNOWN, id, info);
	NodeVec::global()->set_nodevec(TYPE_UNKNOWN, id, info);

	*res_id = NodeVec::global()->index();
}

extern "C" {
	void c_new_node_sub_node_simple(int *ra, int *rb, int *rc, int* res, int *id1) {
		new_node_sub_simple(ra[0], ra[1], rb[0], rb[1], rc[0], rc[1], res, id1);
	}
	void c_new_node_slice_node_simple(int *k, int* res_id, int* id) {
		new_node_slice_simple(*k, res_id, id);
	}
}
