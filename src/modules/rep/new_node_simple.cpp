
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../tree_tool/NodeVec.hpp"

//sqx modify

void new_node_rep_simple(int *res, int *id1 ,int *id2) 
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_REP, id1, id2);

	*res = NodeVec::global()->index();
	return;
}

extern "C" {

	void c_new_node_rep_simple(int *res, int *id1, int* id2)
	{
		new_node_rep_simple(res, id1, id2);
	}
}

