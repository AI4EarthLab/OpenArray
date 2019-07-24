  
  





#include "new_node.hpp"

namespace oa{
  namespace ops{
    NodePtr new_node_plus(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_PLUS);
      np->add_input(0, u);
      np->add_input(1, v);
      
      // const NodeDesc &nd = get_node_desc(TYPE_PLUS);

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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();

      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      // np->display();
      return np;
    }
    NodePtr new_node_minus(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_MINUS);
      np->add_input(0, u);
      np->add_input(1, v);
      
      // const NodeDesc &nd = get_node_desc(TYPE_MINUS);

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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();

      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      // np->display();
      return np;
    }
    NodePtr new_node_mult(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_MULT);
      np->add_input(0, u);
      np->add_input(1, v);
      
      // const NodeDesc &nd = get_node_desc(TYPE_MULT);

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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();

      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      // np->display();
      return np;
    }
    NodePtr new_node_divd(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DIVD);
      np->add_input(0, u);
      np->add_input(1, v);
      
      // const NodeDesc &nd = get_node_desc(TYPE_DIVD);

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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();

      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      // np->display();
      return np;
    }
    NodePtr new_node_pow(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_POW);
      np->add_input(0, u);
      np->add_input(1, v);
      
      // const NodeDesc &nd = get_node_desc(TYPE_POW);

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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();

      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      // np->display();
      return np;
    }

    NodePtr new_node_gt(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_GT);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_ge(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_GE);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_lt(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_LT);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_le(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_LE);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_eq(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_EQ);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_ne(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_NE);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_or(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_OR);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }
    NodePtr new_node_and(const NodePtr& u,
                               const NodePtr& v){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AND);
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

      np->set_bitset(u->get_bitset() | v->get_bitset());

      // split into two fusion kernel when the size of data list is too large
      //int data_list_size = u->get_data_list_size() + v->get_data_list_size();
      //
      //if (data_list_size > 10) {
      //  np->set_lbound({{0,0,0}});
      //  np->set_rbound({{0,0,0}});
      //  u->set_update();
      //  v->set_update();
      //  np->set_data_list_size(1 + 1);
      //} else {
      //  np->set_data_list_size(data_list_size);
      //}

      return np;
    }


    NodePtr new_node_exp(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_EXP);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_sin(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SIN);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_tan(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_TAN);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_cos(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_COS);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_rcp(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_RCP);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_sqrt(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SQRT);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_asin(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_ASIN);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_acos(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_ACOS);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_atan(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_ATAN);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_abs(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_ABS);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_log(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_LOG);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_uplus(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_UPLUS);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_uminus(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_UMINUS);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_log10(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_LOG10);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_tanh(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_TANH);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_sinh(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_SINH);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_cosh(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_COSH);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(u->get_data_type());
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
    NodePtr new_node_not(const NodePtr& u) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_NOT);
      np->add_input(0, u);
      
      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());

      np->set_shape(u->shape());

      np->set_data_type(DATA_INT);
      
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());

      //np->set_data_list_size(u->get_data_list_size());

      return np;
    }
  }
}
