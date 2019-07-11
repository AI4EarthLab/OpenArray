
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../tree_tool/NodeVec.hpp"

//sqx modify


void new_node_shift_simple(int *res, int *id1, int *id2)
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_SHIFT, id1, id2);

	*res = NodeVec::global()->index();
	return;

}

void new_node_circshift_simple(int *res, int *id1, int *id2)
{

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_CIRCSHIFT, id1, id2);

	*res = NodeVec::global()->index();
	return;

}



extern "C"{


void c_new_node_shift_simple(int *res, int *id1, int *id2)
{
	new_node_shift_simple(res, id1, id2);
}

void c_new_node_circshift_simple(int *res, int *id1, int *id2)
{
	new_node_circshift_simple(res, id1, id2);
}
}
