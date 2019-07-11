
#include "../../common.hpp"
#include "../../op_define.hpp"

extern "C"{
  void c_new_node_shift(NodePtr*& o, NodePtr*& u, NodePtr*& v){
    if(o == NULL) o = new NodePtr();
    *o = SHIFT(*u, *v);
  }

  void c_new_node_circshift(NodePtr*& o, NodePtr*& u, NodePtr*& v){
    if(o == NULL) o = new NodePtr();
    *o = CIRCSHIFT(*u, *v);
  }  
}
