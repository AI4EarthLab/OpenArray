#ifndef __SET_NEW_NODE_HPP__
#define __SET_NEW_NODE_HPP__

#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Operator.hpp"
#include "../../utils/utils.hpp"

namespace oa {
  namespace ops {
    NodePtr new_node_set(const NodePtr& u, const NodePtr& v);
  }
}


#endif
