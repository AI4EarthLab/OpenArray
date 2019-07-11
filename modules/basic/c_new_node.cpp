
  
  





#include "../../NodePool.hpp"
#include "../../c-interface/c_oa_type.hpp"

#include "new_node.hpp"

using namespace oa::ops;

extern "C"{
  void c_new_node_plus(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_plus(*u, *v);
  }
  void c_new_node_minus(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_minus(*u, *v);
  }
  void c_new_node_mult(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_mult(*u, *v);
  }
  void c_new_node_divd(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_divd(*u, *v);
  }
  void c_new_node_gt(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_gt(*u, *v);
  }
  void c_new_node_ge(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_ge(*u, *v);
  }
  void c_new_node_lt(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_lt(*u, *v);
  }
  void c_new_node_le(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_le(*u, *v);
  }
  void c_new_node_eq(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_eq(*u, *v);
  }
  void c_new_node_ne(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_ne(*u, *v);
  }
  void c_new_node_pow(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_pow(*u, *v);
  }
  void c_new_node_or(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_or(*u, *v);
  }
  void c_new_node_and(NodePtr*& A,
          const NodePtr*& u, const NodePtr*& v){
    if(A == NULL) A = new NodePtr();
    *A = new_node_and(*u, *v);
  }

  void c_new_node_exp(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_exp(*u);
  }
  void c_new_node_sin(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_sin(*u);
  }
  void c_new_node_tan(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_tan(*u);
  }
  void c_new_node_cos(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_cos(*u);
  }
  void c_new_node_rcp(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_rcp(*u);
  }
  void c_new_node_sqrt(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_sqrt(*u);
  }
  void c_new_node_asin(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_asin(*u);
  }
  void c_new_node_acos(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_acos(*u);
  }
  void c_new_node_atan(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_atan(*u);
  }
  void c_new_node_abs(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_abs(*u);
  }
  void c_new_node_log(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_log(*u);
  }
  void c_new_node_uplus(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_uplus(*u);
  }
  void c_new_node_uminus(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_uminus(*u);
  }
  void c_new_node_log10(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_log10(*u);
  }
  void c_new_node_tanh(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_tanh(*u);
  }
  void c_new_node_sinh(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_sinh(*u);
  }
  void c_new_node_cosh(NodePtr*& A, const NodePtr*& u){
    if(A == NULL) A = new NodePtr();
    *A = new_node_cosh(*u);
  }
}
