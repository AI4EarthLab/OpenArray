#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__ 

#include "NodePool.hpp"
#include "NodeDesc.hpp"

namespace oa {
  namespace ops {

    template<class T>
    NodePtr new_seq_scalar_node(MPI_Comm comm, T val){
      return(NodePool::global()->get_seq_scalar(comm, val));
    }
    
    NodePtr new_node(const ArrayPtr &ap);

    NodePtr new_node(NodeType type, NodePtr u, NodePtr v);

    NodePtr new_node(NodeType type, NodePtr u);

    const NodeDesc& get_node_desc(NodeType type);

    void write_graph(const NodePtr& root);
  }
}


#endif
