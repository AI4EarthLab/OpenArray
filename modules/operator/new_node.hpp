
#ifndef __OP_NEW_NODE_HPP__
#define __OP_NEW_NODE_HPP__

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa{
  namespace ops{

    ///:for o in [i for i in L if i[3] == 'D']
    ///:set name = o[1]
    NodePtr new_node_${name}$(const NodePtr& u);
    ///:endfor
  }
}

#endif
