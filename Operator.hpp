#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__ 

#include "NodePool.hpp"
#include "NodeDesc.hpp"

#define LIB_KERNEL_PATH "./libkernel.so"

namespace oa {
  namespace ops {

    template<class T>
    NodePtr new_seqs_scalar_node(MPI_Comm comm, T val){
      return(NodePool::global()->get_seqs_scalar(comm, val));
    }
    
    NodePtr new_node(const ArrayPtr &ap);

    NodePtr new_node(NodeType type, NodePtr u, NodePtr v);

    NodePtr new_node(NodeType type, NodePtr u);

    const NodeDesc& get_node_desc(NodeType type);

    void write_graph(const NodePtr& root, bool is_root = true,
                     const char *filename = "graph.dot");

    ArrayPtr eval(NodePtr A);

    const KernelPtr get_kernel_dict(size_t hash, 
                                    const char *filename = "fusion-kernels");

    void insert_kernel_dict(size_t hash, const stringstream &s,
                            const char *filename = "fusion-kernels");

    void gen_kernels(NodePtr A, bool is_root = true, MPI_Comm = MPI_COMM_WORLD);

    void tree_to_string(NodePtr A, stringstream &ss);

    void tree_to_code(NodePtr A, stringstream &ss, int &id);

    void tree_to_string_stack(NodePtr A, stringstream &ss);
  }
}


#endif
