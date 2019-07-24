
  
  





#include "../../NodePool.hpp"
#include "../../Grid.hpp"

namespace oa{
  namespace ops{

    NodePtr new_node_dxc(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DXC);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{1, 0, 0}};
      rb = {{1, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DXC));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dyc(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DYC);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 1, 0}};
      rb = {{0, 1, 1}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DYC));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dzc(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DZC);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 1}};
      rb = {{0, 0, 1}};  
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DZC));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_axb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AXB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{1, 0, 0}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AXB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_axf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AXF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{1, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AXF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_ayb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AYB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 1, 0}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AYB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_ayf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AYF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{0, 1, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AYF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_azb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AZB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 1}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AZB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_azf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_AZF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{0, 0, 1}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_AZF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dxb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DXB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{1, 0, 0}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DXB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dxf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DXF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{1, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DXF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dyb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DYB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 1, 0}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DYB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dyf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DYF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{0, 1, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DYF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dzb(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DZB);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 1}};
      rb = {{0, 0, 0}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DZB));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }
    NodePtr new_node_dzf(const NodePtr& u){
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DZF);
      np->add_input(0, u);
      
      int dt = u->get_data_type();

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      np->set_depth(u->get_depth());
      np->set_shape(u->shape());

      if(dt == DATA_INT) dt = DATA_FLOAT;
      
      np->set_data_type(dt);
      np->set_lbound(u->get_lbound());
      np->set_rbound(u->get_rbound());
        
      oa_int3 lb, rb;
    
      lb = {{0, 0, 0}};
      rb = {{0, 0, 1}};
      
      //int grid_data_size = 0;
      /////:if name[0] == 'd'
      //grid_data_size = 1;
      /////:endif


      oa_int3 new_lb = u->get_lbound();
      oa_int3 new_rb = u->get_rbound();
          
      int mx = 0;
      for (int i = 0; i < 3; i++) {
        new_lb[i] += lb[i];
        mx = max(new_lb[i], mx);
        new_rb[i] += rb[i];
        mx = max(new_rb[i], mx);
      }

      if(u->get_pos() != -1) {
        np->set_pos(Grid::global()->get_pos(u->get_pos(), TYPE_DZF));
      }
      /* 
      else {
        grid_data_size = 0;
      }
      */

      //int data_list_size = 2 * u->get_data_list_size() + grid_data_size;

      // set default max stencil as two
      // split into two fusion kernel when the size of data list is too large
      if (mx > 1 /*|| data_list_size > 10*/) {
        np->set_lbound(lb);
        np->set_rbound(rb);
        u->set_update();
        //np->set_data_list_size(2 + grid_data_size);
      } else {
        np->set_lbound(new_lb);
        np->set_rbound(new_rb);
        //np->set_data_list_size(data_list_size);
      }

      np->set_pseudo(u->is_pseudo());

      np->set_bitset(u->get_bitset());
  
      return np;
    }

  }
}
