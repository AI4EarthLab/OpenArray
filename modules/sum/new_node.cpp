#include "new_node.hpp"

namespace oa {
  namespace ops {
    NodePtr new_node_sum(const NodePtr& u, const NodePtr& v) {
      
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SUM);
      np->add_input(0, u);
      np->add_input(1, v);
      
      np->set_lbound({{0, 0, 0}});
      np->set_rbound({{0, 0, 0}});
      np->set_update();
      // u->set_update();
      np->set_data_type(u->get_data_type());
      
      np->set_pos(u->get_pos());

      Shape sp = u->shape();
#ifdef __HAVE_CUDA__
      v->get_data()->memcopy_gpu_to_cpu(); 
      int sum_direct = ((int*)(v->get_data()->get_cpu_buffer()))[0];
#else
      int sum_direct = ((int*)(v->get_data()->get_buffer()))[0];
#endif
      if (sum_direct == 0) sp = {{1, 1, 1}};
      else sp[sum_direct - 1] = 1;
      np->set_shape(sp);

      np->set_bitset();

      //np->set_data_list_size(1);

      return np;
    }

    NodePtr new_node_csum(const NodePtr& u, const NodePtr& v) {
      
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_CSUM);
      np->add_input(0, u);
      np->add_input(1, v);
      
      np->set_lbound({{0, 0, 0}});
      np->set_rbound({{0, 0, 0}});
      np->set_update();
      // u->set_update();
      np->set_data_type(u->get_data_type());
      
      np->set_pos(u->get_pos());
      np->set_shape(u->shape());
      np->set_bitset();

      //np->set_data_list_size(1);

      return np;
    }

  }
}
