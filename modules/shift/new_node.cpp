
#include "../../NodePool.hpp"
#include "../../Grid.hpp"

namespace oa{
  namespace ops{

    
    NodePtr new_node_shift(
        NodePtr& u, NodePtr& v){
    
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SHIFT);
      np->add_input(0, u);
      np->add_input(1, v);
    
      // only OP will change grid pos
      np->set_pos((u)->get_pos());

      np->set_depth((u)->get_depth());
      np->set_shape((u)->shape());

      np->set_data_type((u)->get_data_type());
      np->set_lbound((u)->get_lbound());
      np->set_rbound((u)->get_rbound());
      
      return np;
    }

    NodePtr new_node_circshift(
        NodePtr& u, NodePtr& v){
    
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_CIRCSHIFT);
      np->add_input(0, u);
      np->add_input(1, v);
    
      // only OP will change grid pos
      np->set_pos((u)->get_pos());

      np->set_depth((u)->get_depth());
      np->set_shape((u)->shape());

      np->set_data_type((u)->get_data_type());
      np->set_lbound((u)->get_lbound());
      np->set_rbound((u)->get_rbound());
      
      return np;
    }
  }
}
