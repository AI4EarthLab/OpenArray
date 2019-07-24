
  
  





#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
void c_new_node_dxc_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DXC, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dyc_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DYC, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dzc_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DZC, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_axb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AXB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_axf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AXF, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_ayb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AYB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_ayf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AYF, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_azb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AZB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_azf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_AZF, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dxb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DXB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dxf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DXF, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dyb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DYB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dyf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DYF, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dzb_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DZB, id1);

	*res = NodeVec::global()->index();
}
void c_new_node_dzf_simple(int *res, int *id1){

	NodeVec::global()->set_opnode_size();
	int opnode_size = NodeVec::global()->get_opnode_size();
	int nodevec_size = NodeVec::global()->get_nodevec_size();

	if(opnode_size > nodevec_size) 
		NodeVec::global()->reset_nodevec();

	NodeVec::global()->set_nodevec(TYPE_DZF, id1);

	*res = NodeVec::global()->index();
}
}
