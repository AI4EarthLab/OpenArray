
#ifndef __NODEPOOL_HPP__
#define __NODEPOOL_HPP__
#include <list>
#include "common.hpp"
#include "Node.hpp"
#include "ArrayPool.hpp"

class NodePool{
  typedef std::list<Node*> NodeList;
  NodeList m_list;

public:

  //get an object from the pool
  NodePtr get(){
    Node *p = NULL;
    if(m_list.size() > 0){
      p = m_list.back();
      m_list.pop_back();
    }else{
      p = new Node();
    }
    
    return NodePtr(p, [](Node* np){
	NodePool::global()->dispose(np);
      });
  }

  template<class T>
  NodePtr get_seq_scalar(T val){
    NodePtr p = NodePool::global()->get();
    ArrayPtr ap = ArrayPool::global()->get_seq_scalar(val);
    p->set_type(TYPE_DATA);
    p->set_data(ap);
  }
  
  // throw the object into object pool 
  void dispose(Node* n){
    n -> reset();
    m_list.push_back(n);
  }

  static NodePool* global(){
    static NodePool np;
    return &np;
  }
};

#endif
