
#include "../../common.hpp"
#include "../../NodePool.hpp"


NodePtr new_node_max (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_MAX);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(SCALAR_SHAPE);

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_max (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_max (*p2);
  }
}


NodePtr new_node_max_at (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_MAX_AT);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(Shape({{3,1,1}}));

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_max_at (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_max_at (*p2);
  }
}


NodePtr new_node_min (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_MIN);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(SCALAR_SHAPE);

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_min (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_min (*p2);
  }
}


NodePtr new_node_min_at (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_MIN_AT);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(Shape({{3,1,1}}));

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_min_at (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_min_at (*p2);
  }
}


NodePtr new_node_abs_max (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_ABS_MAX);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(SCALAR_SHAPE);

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_abs_max (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_abs_max (*p2);
  }
}


NodePtr new_node_abs_max_at (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_ABS_MAX_AT);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(Shape({{3,1,1}}));

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_abs_max_at (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_abs_max_at (*p2);
  }
}


NodePtr new_node_abs_min (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_ABS_MIN);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(SCALAR_SHAPE);

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_abs_min (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_abs_min (*p2);
  }
}


NodePtr new_node_abs_min_at (const NodePtr& p)
{
  NodePtr np = NodePool::global()->get();
  np->set_type(TYPE_ABS_MIN_AT);
  np->set_data_type(p->get_data_type());  
  np->add_input(0, p);

  np->set_shape(Shape({{3,1,1}}));

  //np->set_data_list_size(1);
  
  return np;
}

extern "C"{
  void c_new_node_abs_min_at (NodePtr*& p1, NodePtr*& p2)
  {
    if(p1 == NULL) p1 = new NodePtr();
  
    *p1 = new_node_abs_min_at (*p2);
  }
}









