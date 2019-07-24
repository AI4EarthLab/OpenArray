
  
  





#include "TreeRootDict.hpp"
#include <stdio.h>
#include <iostream>
#include "Operator.hpp"
#include <cstdlib>
using namespace oa::ops;
vector<NodePtr> data_nodes_vec;
NodePtr& oa_build_tree()
{
    auto &simp_nodes_vec = NodeVec::global()->get_ndptr_s();
    auto &simp_data_nodes_vec = NodeVec::global()->get_datand_s();
    size_t hash = NodeVec::global()->get_hash();
    int simp_nodesize = NodeVec::global()->get_opnode_size();
    bool found = TreeRootDict::global()->find(hash);
    if (false == found)
    {
        data_nodes_vec.clear();
        for (int i = 0; i < simp_nodesize; i++)
        {
            auto &sn = simp_nodes_vec[i];
            int node_type = sn.type;
            if (node_type == TYPE_DATA)
            {
                //update ArrayPtr
                NodePtr new_op = new_node(sn.get_ArrayPtr());
                sn.ndp = new_op;
                data_nodes_vec.push_back(new_op);
            }

            else if (node_type == TYPE_MIN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_min(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MAX)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_max(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MIN_AT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_min_at(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MAX_AT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_max_at(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ABS_MAX)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_abs_max(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ABS_MIN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_abs_min(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ABS_MAX_AT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_abs_max_at(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ABS_MIN_AT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_abs_min_at(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_EXP)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_exp(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SIN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_sin(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_TAN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_tan(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_COS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_cos(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_RCP)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_rcp(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SQRT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_sqrt(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ASIN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_asin(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ACOS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_acos(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ATAN)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_atan(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_ABS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_abs(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_LOG)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_log(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_UPLUS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_uplus(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_UMINUS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_uminus(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_LOG10)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_log10(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_TANH)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_tanh(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SINH)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_sinh(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_COSH)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_cosh(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DXC)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dxc(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DYC)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dyc(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DZC)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dzc(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AXB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_axb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AXF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_axf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AYB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_ayb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AYF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_ayf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AZB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_azb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AZF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_azf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DXB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dxb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DXF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dxf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DYB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dyb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DYF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dyf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DZB)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dzb(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DZF)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_dzf(u);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_NOT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_not(u);
                sn.ndp = new_op;
            }

            else if (node_type == TYPE_PLUS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_plus(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MINUS)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_minus(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MULT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_mult(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_DIVD)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_divd(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_GT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_gt(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_GE)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_ge(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_LT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_lt(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_LE)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_le(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_EQ)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_eq(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_NE)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_ne(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_POW)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_pow(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SUM)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_sum(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_CSUM)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_csum(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_OR)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_or(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_AND)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_and(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SHIFT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_shift(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_CIRCSHIFT)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_circshift(u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_SET)
            {
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node_set(u, v);
                sn.ndp = new_op;
            }

            else if (node_type == TYPE_MIN2)
            {

                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node((NodeType)node_type, u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_MAX2)
            {

                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node((NodeType)node_type, u, v);
                sn.ndp = new_op;
            }
            else if (node_type == TYPE_REP)
            {

                auto &input_0 = simp_nodes_vec[sn.input[0]];
                auto &input_1 = simp_nodes_vec[sn.input[1]];
                NodePtr &u = input_0.ndp;
                NodePtr &v = input_1.ndp;
                NodePtr new_op = new_node((NodeType)node_type, u, v);
                sn.ndp = new_op;
            }

            else if (node_type == TYPE_INT || node_type == TYPE_FLOAT || node_type == TYPE_DOUBLE)
            {
                NodePtr new_op = NULL;
                switch(node_type) {
                    case TYPE_INT:
                    new_op  = new_seqs_scalar_node(*((int*)sn.get_val()));
                    break;
                    case TYPE_FLOAT:
                    new_op  = new_seqs_scalar_node(*((float*)sn.get_val()));
                    break;
                    case TYPE_DOUBLE:
                    new_op  = new_seqs_scalar_node(*((double*)sn.get_val()));
                    break;

                }
                sn.ndp = new_op;
                data_nodes_vec.push_back(new_op);
            }
            //slice node
            else if(node_type  == TYPE_UNKNOWN){
                auto &input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                NodePtr new_op = new_node_slice(u, *((int *)sn.get_val()));
                data_nodes_vec.push_back(new_op);
		        sn.ndp = new_op;
            }

            //type of sub node is type_ref
            else if (node_type == TYPE_REF)
            {
                auto& input_0 = simp_nodes_vec[sn.input[0]];
                NodePtr &u = input_0.ndp;
                int *box = (int *)sn.get_val();
                NodePtr new_op = new_node_sub(u, Box(box[0], box[1], box[2], box[3], box[4], box[5]));
                sn.ndp = new_op;
                //need set ref every time
                data_nodes_vec.push_back(new_op);
            }

            else if (node_type == TYPE_INT3_REP || node_type == TYPE_INT3_SHIFT)
            {
                NodePtr new_op = NodePool::global()->get_local_1d<int, 3>((int *)sn.get_val());
                sn.ndp = new_op;
                data_nodes_vec.push_back(new_op);
            }

            else
            {
                std::cout << "==============Get wrong node type when building tree, exit!==================\n";
                std::cout << "Node Type: " << node_type << endl;
                exit(EXIT_FAILURE);
            }
        }
        //NodePtr& root_node = simp_nodes_vec[simp_nodes_vec.size()-1].ndp;
        NodePtr &root_node = simp_nodes_vec[simp_nodesize - 1].ndp;

        TreeRootDict::global()->insert(hash, root_node);
        TreeDataNodes::global()->insert(hash, data_nodes_vec);
	if(root_node->type() == TYPE_SET){
	        NodePtr A = root_node->input(0);
	        NodePtr B = root_node->input(1);
		 if (false == A->is_ref()) {
      			oa::ops::gen_kernels_JIT_with_op(A);
		}
		 if (false == B->is_ref()) {
      			oa::ops::gen_kernels_JIT_with_op(B);
    }
	}
	else 
        	oa::ops::gen_kernels_JIT_with_op(root_node, true);
        data_nodes_vec.clear();
    }

    //auto& cache_data_nodes_vec = TreeDataNodes::global()->get(hash);

    //updata data nodes
    else 
        TreeDataNodes::global()->modify(hash, simp_data_nodes_vec);

    //NodeVec::global()->clear();  
		NodeVec::global()->set_nodesize_zero();
    /* 
    NodePtr tmp = TreeRootDict::global()->get(hash);
    tmp->display(); 
    for(int i=0;i<tmp->input_size();i++){
        tmp->input(i)->display();
    }*/
    return TreeRootDict::global()->get(hash);
}

void data_vec_clear(){
 data_nodes_vec.clear();
}

void tree_clear(){
	TreeRootDict::global()->clear();
	TreeDataNodes::global()->clear();
        data_vec_clear();
	NodeVec::global()->clear();  
}
