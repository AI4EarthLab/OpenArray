
///:include "../../NodeType.fypp"
#include "../../NodePool.hpp"
#include "../../Grid.hpp"

namespace oa{
  namespace ops{

    NodePtr new_node_sub(NodePtr& u, const Box& b){

      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_REF);
      np->add_input(0, u);
      np->set_ref(b);

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(b.shape());

      np->set_data_type(u->get_data_type());
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
      
      np->set_bitset();

      np->set_data_list_size(1);
      
      return np;
    }

    NodePtr new_node_sub(ArrayPtr& u, const Box& b){

      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_REF);
      np->set_data(u);
      np->set_ref(b);

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(0);
      np->set_shape(b.shape());

      np->set_data_type(u->get_data_type());

      np->set_bitset();

      np->set_data_list_size(1);

      return np;
    }

    NodePtr new_node_slice(NodePtr& u, int k){

      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_UNKNOWN);
      np->add_input(0, u);
      np->set_slice(k);

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      
      Shape sp = u->shape();
      sp[2] = 1;
      np->set_shape(sp);

      np->set_data_type(u->get_data_type());
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_bitset();

      return np;
    }

    NodePtr new_node_slice(ArrayPtr& u, int k){

      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_UNKNOWN);
      np->set_data(u);
      np->set_slice(k);

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(0);
      
      Shape sp = u->shape();
      sp[2] = 1;
      np->set_shape(sp);

      np->set_data_type(u->get_data_type());

      np->set_bitset();

      return np;
    }

  }
}
