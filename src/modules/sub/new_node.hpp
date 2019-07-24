
#ifndef __SUB_NEW_NODE_HPP__
#define __SUB_NEW_NODE_HPP__


namespace oa{
  namespace ops{
    NodePtr new_node_sub(NodePtr& u, const Box& b);
    NodePtr new_node_sub(ArrayPtr& u, const Box& b);
    NodePtr new_node_slice(NodePtr& u, int k);
    NodePtr new_node_slice(ArrayPtr& u, int k);
  }
}

#endif
