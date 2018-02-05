
#ifndef __NODEPOOL_HPP__
#define __NODEPOOL_HPP__
#include <list>
#include "common.hpp"
#include "Node.hpp"
#include "ArrayPool.hpp"
#include "Function.hpp"

class NodePool {
  typedef std::list<Node*> NodeList;
  NodeList m_list;
  int global_count = 0;

public:

  //get an object from the pool
  NodePtr get() {
    
    Node *p = NULL;
    if(m_list.size() > 0) {
      p = m_list.back();
      m_list.pop_back();
    }else{
      p = new Node();
      add_count();
      if (g_debug) cout<<"NodePool.size() = "<<count()<<endl;
    }
    p->set_id(NodePool::global_id());

    return NodePtr(p, [](Node* np) {
        NodePool::global()->dispose(np);
      });
  }

  template<class T>
  NodePtr get_seqs_scalar(T val) {
    NodePtr p = NodePool::global()->get();
    ArrayPtr ap = oa::funcs::consts(MPI_COMM_SELF,
            {{1, 1, 1}}, val, 0);
    p->clear_input();
    p->set_type(TYPE_DATA);
    p->set_data(ap);
    p->set_scalar();
    p->set_seqs();
    p->set_data_type(ap->get_data_type());
    p->set_shape(ap->shape());
    p->set_bitset(ap->get_bitset());
    p->set_pseudo(ap->is_pseudo());
    return p;
  }

  template<class T, int size>
  NodePtr get_local_1d(T* val) {
    NodePtr p = NodePool::global()->get();
    ArrayPtr ap = oa::funcs::consts(MPI_COMM_SELF,
            {{size,1,1}}, 0, DATA_INT);
    oa::internal::copy_buffer((T*)ap->get_buffer(), val, size);
    p->clear_input();
    p->set_type(TYPE_DATA);
    p->set_data(ap);
    p->set_seqs();
    p->set_data_type(ap->get_data_type());
    p->set_shape(ap->shape());
    p->set_bitset(ap->get_bitset());
    p->set_pseudo(ap->is_pseudo());
    return p;
  }
  
  // throw the object into object pool 
  void dispose(Node* n) {
    if (n == NULL) return ;
    n -> reset();
    m_list.push_back(n);
  }

  static NodePool* global() {
    static NodePool np;
    return &np;
  }

  static int global_id() {
    static int m_global_id = 0;
    m_global_id += 1;
    return m_global_id;
  }

  int count() {
    return global_count;
  }

  void add_count() {
    global_count += 1;
  }

};

#endif
