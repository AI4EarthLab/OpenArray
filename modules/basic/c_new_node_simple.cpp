
  
  





#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  void c_new_node_plus_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_PLUS, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_minus_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_MINUS, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_mult_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_MULT, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_divd_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_DIVD, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_gt_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_GT, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_ge_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_GE, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_lt_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_LT, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_le_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_LE, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_eq_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_EQ, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_ne_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_NE, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_pow_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_POW, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_or_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_OR, id1, id2);

		*res = NodeVec::global()->index();
  }
  void c_new_node_and_simple(int *res, int *id1, int *id2){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_AND, id1, id2);

		*res = NodeVec::global()->index();
  }

  void c_new_node_exp_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_EXP, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_sin_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_SIN, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_tan_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_TAN, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_cos_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_COS, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_rcp_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_RCP, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_sqrt_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_SQRT, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_asin_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_ASIN, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_acos_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_ACOS, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_atan_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_ATAN, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_abs_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_ABS, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_log_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_LOG, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_uplus_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_UPLUS, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_uminus_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_UMINUS, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_log10_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_LOG10, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_tanh_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_TANH, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_sinh_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_SINH, id1);

		*res = NodeVec::global()->index();
  }
  void c_new_node_cosh_simple(int *res, int *id1){
    //xiaogang

		NodeVec::global()->set_opnode_size();
		int opnode_size = NodeVec::global()->get_opnode_size();
		int nodevec_size = NodeVec::global()->get_nodevec_size();

		if(opnode_size > nodevec_size) 
			NodeVec::global()->reset_nodevec();

		NodeVec::global()->set_nodevec(TYPE_COSH, id1);

		*res = NodeVec::global()->index();
  }
}
