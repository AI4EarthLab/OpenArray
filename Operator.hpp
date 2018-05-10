/*
 * Operator.hpp
 * evaluate the expression graph
 *
=======================================================*/

#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__ 

#include "NodePool.hpp"
#include "NodeDesc.hpp"

#define LIB_KERNEL_PATH "./libkernel.so"

namespace oa {
  namespace ops {

    // create a new sequences scalar node
    template<class T>
    NodePtr new_seqs_scalar_node(T val){
      return(NodePool::global()->get_seqs_scalar(val));
    }
    
    // create a new node based on ArrayPtr ap
    NodePtr new_node(const ArrayPtr &ap);

    // create a new node on binaray operator 
    // which NodeType is type & input is u and v
    NodePtr new_node(NodeType type, NodePtr u, NodePtr v);

    //! get description of an operator for a given type
    const NodeDesc& get_node_desc(NodeType type);

    // write the expression graph into graph.dot
    void write_graph(const NodePtr& root, bool is_root = true,
                     const char *filename = "graph.dot");

    // force eval expression graph, eval without fusion kernels
    ArrayPtr force_eval(NodePtr A);

    // prepare kernel fusion parameters with operator
    void get_kernel_parameter_with_op(NodePtr A, vector<void*> &list, 
      vector<ArrayPtr> &update_list, vector<int3> &S, PartitionPtr &ptr, 
      bitset<3> &bt, vector<int3> &lb_list, vector<int3> &rb_list,
      int3 lb_now, int3 rb_now, vector<ArrayPtr> &data_list); 

    // eval the expression graph by using fusion kernels
    ArrayPtr eval(NodePtr A);

    // generate fusion kernels
    void gen_kernels_JIT_with_op(NodePtr A, 
        bool is_root = true);

    void tree_to_string(NodePtr A, stringstream &ss);

    void tree_to_code_with_op(NodePtr A, stringstream &ss, stringstream &__point, int &id, int &S_id,
      vector<int>& int_id, vector<int>& float_id, vector<int>& double_id);

    void change_string_with_op(stringstream& ss, string in, const NodeDesc &nd);

    string replace_string(string& in, const string& old_str, const string& new_str);

    void tree_to_string_stack(NodePtr A, stringstream &ss);

    void code_add_function_signature(stringstream& code, size_t& hash);

    void code_add_function_signature_with_op(stringstream& code, size_t& hash);

    void code_add_const(stringstream& code, 
        vector<int>& i, vector<int>& f, vector<int>& d);
    
    void code_add_function(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id);

    void code_add_calc_outside(stringstream& code, 
      stringstream& __code, stringstream& __point, DATA_TYPE dt, int& id, int& S_id);

    void code_add_calc_inside(stringstream& code, 
      stringstream& __code, stringstream& __point, DATA_TYPE dt, int& id, int& S_id);    
  }
}


#endif
