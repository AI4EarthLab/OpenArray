#include "new_node.hpp"

namespace oa{
  namespace ops{
    NodePtr new_node_set(const NodePtr& u,const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SET);
      np->add_input(0, u);
      np->add_input(1, v);
     return np;
    }
  }
} 
