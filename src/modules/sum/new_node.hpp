#ifndef __SUM_NEW_NODE_HPP__
#define __SUM_NEW_NODE_HPP__

#include "../../NodePool.hpp"

namespace oa {
  namespace ops {
    NodePtr new_node_sum(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_csum(const NodePtr& u, const NodePtr& v);
  }
}

#endif