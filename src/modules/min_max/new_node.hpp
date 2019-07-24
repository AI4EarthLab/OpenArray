
#include "../../common.hpp"
#include "../../NodePool.hpp"


NodePtr new_node_max (const NodePtr& p);

extern "C"{
  void c_new_node_max (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_max_at (const NodePtr& p);

extern "C"{
  void c_new_node_max_at (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_min (const NodePtr& p);

extern "C"{
  void c_new_node_min (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_min_at (const NodePtr& p);

extern "C"{
  void c_new_node_min_at (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_abs_max (const NodePtr& p);

extern "C"{
  void c_new_node_abs_max (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_abs_max_at (const NodePtr& p);

extern "C"{
  void c_new_node_abs_max_at (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_abs_min (const NodePtr& p);

extern "C"{
  void c_new_node_abs_min (NodePtr*& p1, NodePtr*& p2);
}


NodePtr new_node_abs_min_at (const NodePtr& p);

extern "C"{
  void c_new_node_abs_min_at (NodePtr*& p1, NodePtr*& p2);
}









