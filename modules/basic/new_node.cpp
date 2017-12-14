///:include "../../NodeType.fypp"
#include "new_node.hpp"

namespace oa{
  namespace ops{
    ///:for op in [i for i in L if i[3] == 'A']
    NodePtr new_node_${op[1]}$(NodeType type,
                               const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      
      const NodeDesc &nd = get_node_desc(type);

      DataType dt = oa::utils::cast_data_type(u->get_data_type(),
                                              v->get_data_type());

      np->set_depth(u->get_depth(), v->get_depth());
      
      // U and V must have same shape
      if (u->is_seqs_scalar()) np->set_shape(v->shape());
      else if (v->is_seqs_scalar()) np->set_shape(u->shape());
      else {
        /*
          pseudo 3d, ans's should be the bigger one among u'shape & v'shape
        */
        bool flag = false;
        for (int i = 0; i < 3; i++) {
          if (u->shape()[i] > v->shape()[i]) flag = true;
        }
        if (flag) np->set_shape(u->shape());
        else np->set_shape(v->shape());
      }
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound(), v->get_lbound());
      np->set_rbound(u->get_rbound(), v->get_rbound());

      // u & v must in the same grid pos
      //assert(u->get_pos() == v->get_pos());
      if(u->get_pos() != -1)
        np->set_pos(u->get_pos());
      else if(v->get_pos() != -1)
        np->set_pos(v->get_pos());
     
      np->set_pseudo(u->is_pseudo() && v->is_pseudo());

      return np;
    }
    ///:endfor

    ///:for op in [i for i in L if i[3] == 'B']
    NodePtr new_node_${op[1]}$(NodeType type,
                               const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      
      np->set_depth(u->get_depth(), v->get_depth());
      
      // U and V must have same shape
      if (u->is_seqs_scalar()) np->set_shape(v->shape());
      else if (v->is_seqs_scalar()) np->set_shape(u->shape());
      else {
        /*
          pseudo 3d, so don't have to assert
          assert(oa::utils::is_equal_shape(u->shape(), v->shape()));
        */
        np->set_shape(u->shape());
      }
      
      np->set_data_type(DATA_INT);
      
      np->set_lbound(u->get_lbound(), v->get_lbound());
      np->set_rbound(u->get_rbound(), v->get_rbound());

      // u & v must in the same grid pos
      //assert(u->get_pos() == v->get_pos());
      if(u->get_pos() != -1)
        np->set_pos(u->get_pos());
      else if(v->get_pos() != -1)
        np->set_pos(v->get_pos());
      
      np->set_pseudo(u->is_pseudo() && v->is_pseudo());

      return np;
    }
    ///:endfor


    ///:for op in [i for i in L if i[3] == 'C']
    NodePtr new_node_${op[1]}$(NodeType type,
                               const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());
      return np;
    }
    ///:endfor
  }
}
