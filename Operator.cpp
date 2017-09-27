

#include "Node.hpp"

class Operator {
private:
public:
  int greater_than(NodePtr u, NodePtr v);
  int greater_equal(NodePtr u, NodePtr v);
  int less_than(NodePtr u, NodePtr v);
  int less_equal(NodePtr u, NodePtr v);
  int equal(NodePtr u, NodePtr v);
  int not_equal(NodePtr u, NodePtr v);

  
  static NodePtr minus(NodePtr u, NodePtr v);
  static NodePtr mult(NodePtr u, NodePtr v);
  static NodePtr divd(NodePtr u, NodePtr v);

  static NodePtr max(NodePtr u);
  static NodePtr min(NodePtr u);
  static NodePtr pow(NodePtr u, NodePtr v);
  static NodePtr exp(NodePtr u);
  static NodePtr sin(NodePtr u);
  static NodePtr cos(NodePtr u);
  static NodePtr tan(NodePtr u);
  static NodePtr rcp(NodePtr u);
  static NodePtr sqrt(NodePtr u);
  static NodePtr asin(NodePtr u);
  static NodePtr acos(NodePtr u);
  static NodePtr atan(NodePtr u);
  static NodePtr abs(NodePtr u);
  static NodePtr log(NodePtr u);
  static NodePtr uplus(NodePtr u);
  static NodePtr uminus(NodePtr u);
  static NodePtr log10(NodePtr u);
  static NodePtr tanh(NodePtr u);
  static NodePtr sinh(NodePtr u);
  static NodePtr cosh(NodePtr u);
  static NodePtr dxc(NodePtr u);
  static NodePtr dyc(NodePtr u);
  static NodePtr dzc(NodePtr u);
  static NodePtr axb(NodePtr u);
  static NodePtr ayb(NodePtr u);
  static NodePtr azb(NodePtr u);
  static NodePtr axf(NodePtr u);
  static NodePtr ayf(NodePtr u);
  static NodePtr azf(NodePtr u);
  static NodePtr dxb(NodePtr u);
  static NodePtr dyb(NodePtr u);
  static NodePtr dzb(NodePtr u);
  static NodePtr dxf(NodePtr u);
  static NodePtr dyf(NodePtr u);
  static NodePtr dzf(NodePtr u);
  static NodePtr sum(NodePtr u);
  static NodePtr csum(NodePtr u);
  static NodePtr operator_or(NodePtr u);
  static NodePtr operator_and(NodePtr u);
  static NodePtr operator_not(NodePtr u);
  static NodePtr repeat(NodePtr u);
  static NodePtr shift(NodePtr u);
};
