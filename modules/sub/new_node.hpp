
#ifndef __SUB_NEW_NODE_HPP__
#define __SUB_NEW_NODE_HPP__

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa{
  namespace ops{
    NodePtr new_node(NodePtr& u, const Box& b);
    NodePtr new_node(ArrayPtr& u, const Box& b);
  }
}

#endif
