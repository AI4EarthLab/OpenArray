
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../tree_tool/NodeVec.hpp"

//sqx modify


void new_node_sum_simple(int *id_res, int *id1, int *id2) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_SUM, id1, id2);

	*id_res = NodeVec::global()->index();
	return;
}

void new_node_csum_simple(int *id_res, int *id1, int *id2) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_CSUM, id1, id2);

	*id_res = NodeVec::global()->index();
	return;
}

extern "C" {

	void c_new_node_sum_simple(int *id_res, int *id1, int *id2)
	{
		new_node_sum_simple(id_res, id1, id2);
	}

	void c_new_node_csum_simple(int *id_res, int *id1, int *id2)
	{
		new_node_csum_simple(id_res, id1, id2);
	}
}

