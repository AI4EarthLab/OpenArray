
#include "../../common.hpp"
#include "../../NodePool.hpp"

///:for n1 in ['',    'abs_']  
///:for n2 in ['max', 'min']
///:for n3 in ['',    '_at']
///:set name = "{0}{1}{2}".format(n1,n2,n3)

NodePtr new_node_${name}$ (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_${name.upper()}$);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  ///:if (n3 == '_at')
  np->set_shape(Shape({{3,1,1}}));
  ///:else
  np->set_shape(SCALAR_SHAPE);
  ///:endif

  np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_${name}$ (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_${name}$ (*p2);
  }
}

///:endfor
///:endfor
///:endfor








