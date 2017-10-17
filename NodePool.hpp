

#ifndef __NODEPOOL_HPP__
#define __NODEPOOL_HPP__
#include <list>

class NodePool{
  typedef std::list<Node*> NodeList;
  NodeList m_list;
  
public:
  NodePtr get(){
  Node *p = NULL:
  if(m_list.size() > 0){
  p = m_list.back()
  }else{
  p = new Node();
  }
  return NodePtr(p, [](Node* np){
	NodePool::global()-restore(np);
  })
  }

  void restore(Node* n){
  m_list.push_back(n);
  }

  static const NodePool* global(){
  static NodePool np;
  return &np;
  }
};

#endif
