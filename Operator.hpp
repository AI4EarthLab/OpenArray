#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__ 

#include "NodePool.hpp"
#include "NodeDesc.hpp"

namespace oa{
  namespace ops{

    template<class T>
    NodePtr new_seq_scalar_node(T val){
      return(NodePool::global()->get_seq_scalar(val));
    }
    
    NodePtr new_node(NodeType type, NodePtr u, NodePtr v);

    NodePtr new_node(NodeType type, NodePtr u);

    const NodeDesc& get_node_desc(NodeType type);

    void write_graph(const NodePtr& root);
  }
}


#endif
