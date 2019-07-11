
  
  





#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  void c_new_node_dxc(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dxc(*u);
  }
  void c_new_node_dyc(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dyc(*u);
  }
  void c_new_node_dzc(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dzc(*u);
  }
  void c_new_node_axb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_axb(*u);
  }
  void c_new_node_axf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_axf(*u);
  }
  void c_new_node_ayb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_ayb(*u);
  }
  void c_new_node_ayf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_ayf(*u);
  }
  void c_new_node_azb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_azb(*u);
  }
  void c_new_node_azf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_azf(*u);
  }
  void c_new_node_dxb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dxb(*u);
  }
  void c_new_node_dxf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dxf(*u);
  }
  void c_new_node_dyb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dyb(*u);
  }
  void c_new_node_dyf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dyf(*u);
  }
  void c_new_node_dzb(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dzb(*u);
  }
  void c_new_node_dzf(NodePtr*& A, const NodePtr*& u){
    c_destroy_node((void*&)A);

    A = new NodePtr();
    *A = new_node_dzf(*u);
  }
}
