
#include "Operator.hpp"

namespace oa {
  namespace ops{

    NodePtr new_node(const ArrayPtr &ap) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DATA);
      np->set_data(ap);
      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      return np;
    }

    //! get description of an operator for a given type
    const NodeDesc& get_node_desc(NodeType type){

      static bool has_init = false;                                            
      static OpDescList s;
      
      if(!has_init){
	s.resize(NUM_NODE_TYPES);
#:mute
#:set i = 0  
#:include "NodeType.fypp"
#:endmute
	  //intialize node descriptions.
#:for i in L
#:mute
#:set type = i[0]
#:set name = i[1]
#:set ew = i[5]
#:if ew == 'F'
#:set ew = 'false'
#:else
#:set ew = 'true'
#:endif
#:set cl = i[6]
#:if cl == 'F'
#:set cl = 'false'
#:else
#:set cl = 'true'
#:endif
#:endmute
#:set ef = i[7]    
         s[${type}$] = {${type}$, "${name}$", ${ew}$, ${cl}$, "${ef}$"};
#:endfor
	 has_init = true;
      }
      return s.at(type);
    }

    void write_graph(const NodePtr& root){
      
    }
  }
}


