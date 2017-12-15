///:include "../../NodeType.fypp"
#include "../../common.hpp"
#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Operator.hpp"
#include "../../utils/utils.hpp"

namespace oa{
  namespace ops{
    ///:for op in [i for i in L if i[3] == 'A']
    NodePtr new_node_${op[1]}$(const NodePtr& u, const NodePtr& v);
    ///:endfor

    ///:for op in [i for i in L if i[3] in ['B', 'F']]
    NodePtr new_node_${op[1]}$(const NodePtr& u, const NodePtr& v);
    ///:endfor

    ///:for op in [i for i in L if i[3] == 'C']
    NodePtr new_node_${op[1]}$(const NodePtr& u);
    ///:endfor

    NodePtr new_node_not(const NodePtr& u);

    NodePtr new_node_pow(const NodePtr& u, const NodePtr& v);

  }
}
