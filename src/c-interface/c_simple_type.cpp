#include "c_simple_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Kernel.hpp"
#include "../op_define.hpp"
#include "../MPI.hpp"

//xiaogang

extern "C" {

	void c_new_node_array_simple(ArrayPtr* &ap, int *res)
	{
		assert(ap != NULL &&
						"array pointer can not be null to create a node.");
		//oa::ops::new_node_simple(*(ArrayPtr*)ap);

		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_DATA, ap);
		NodeVec::global()->set_nodevec(TYPE_DATA, ap);

		*res = NodeVec::global()->index();
	}

	

	
	void c_new_seqs_scalar_node_int_simple(int val, int *res) {
		// xiaogang

		extra_info info;
		info.union_value_int = val;

		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_INT, info);
		NodeVec::global()->set_nodevec(TYPE_INT, info);

		*res = NodeVec::global()->index();
	}
	
	void c_new_seqs_scalar_node_float_simple(float val, int *res) {
		// xiaogang

		extra_info info;
		info.union_value_float = val;

		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_FLOAT, info);
		NodeVec::global()->set_nodevec(TYPE_FLOAT, info);

		*res = NodeVec::global()->index();
	}
	
	void c_new_seqs_scalar_node_double_simple(double val, int *res) {
		// xiaogang

		extra_info info;
		info.union_value_double = val;

		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_DOUBLE, info);
		NodeVec::global()->set_nodevec(TYPE_DOUBLE, info);

		*res = NodeVec::global()->index();
	}

	//sqx modify

	void c_new_node_int3_simple_rep(int *x, int *y, int *z, int *res) {

		extra_info info;

		info.union_int3[0] = *x;
		info.union_int3[1] = *y;
		info.union_int3[2] = *z;


		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_INT3_REP, info);
		NodeVec::global()->set_nodevec(TYPE_INT3_REP, info);

		*res = NodeVec::global()->index();
	}

	void c_new_node_int3_simple_shift(int *x, int *y, int *z, int *res) {

		extra_info info;

		info.union_int3[0] = *x;
		info.union_int3[1] = *y;
		info.union_int3[2] = *z;


		NodeVec::global()->set_datanode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();
		int datanode_size = NodeVec::global()->get_datanode_size();
		int datavec_size = NodeVec::global()->get_datavec_size();

		if(datanode_size > datavec_size)
			NodeVec::global()->reset_datavec();
		
		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_datavec(TYPE_INT3_SHIFT, info);
		NodeVec::global()->set_nodevec(TYPE_INT3_SHIFT, info);

		*res = NodeVec::global()->index();
	}
	
	void c_new_node_op2_simple(int nodetype, int *res_id, int *id1, int *id2) {

		NodeVec::global()->set_opnode_size();

		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(nodetype, id1, id2);

		*res_id = NodeVec::global()->index();
	}

}











