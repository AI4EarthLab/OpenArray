  
  





#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Operator.hpp"
#include "../../utils/utils.hpp"

namespace oa{
  namespace ops{
    NodePtr new_node_plus(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_minus(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_mult(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_divd(const NodePtr& u, const NodePtr& v);

    NodePtr new_node_gt(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_ge(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_lt(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_le(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_eq(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_ne(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_or(const NodePtr& u, const NodePtr& v);
    NodePtr new_node_and(const NodePtr& u, const NodePtr& v);

    NodePtr new_node_exp(const NodePtr& u);
    NodePtr new_node_sin(const NodePtr& u);
    NodePtr new_node_tan(const NodePtr& u);
    NodePtr new_node_cos(const NodePtr& u);
    NodePtr new_node_rcp(const NodePtr& u);
    NodePtr new_node_sqrt(const NodePtr& u);
    NodePtr new_node_asin(const NodePtr& u);
    NodePtr new_node_acos(const NodePtr& u);
    NodePtr new_node_atan(const NodePtr& u);
    NodePtr new_node_abs(const NodePtr& u);
    NodePtr new_node_log(const NodePtr& u);
    NodePtr new_node_uplus(const NodePtr& u);
    NodePtr new_node_uminus(const NodePtr& u);
    NodePtr new_node_log10(const NodePtr& u);
    NodePtr new_node_tanh(const NodePtr& u);
    NodePtr new_node_sinh(const NodePtr& u);
    NodePtr new_node_cosh(const NodePtr& u);

    NodePtr new_node_not(const NodePtr& u);

    NodePtr new_node_pow(const NodePtr& u, const NodePtr& v);

  }
}
