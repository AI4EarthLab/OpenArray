#ifndef __OPERATOR_HPP__
#define __OPERATOR_HPP__ 

#include "Node.hpp"
#include "NodeDesc.hpp"

namespace oa{
  namespace ops{
    
    NodePtr new_seq_scalar_node(void* val, DataType t){
      return(NodePool::get_seq_scalar(val, t));
    }

    NodePtr new_node(DataType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->type = type;
      np->add_input(0, u);
      np->add_input(1, v);
      return np;
    }

    NodePtr new_node(DataType type, NodePtr u){
      NodePtr np = NodePool::global()->get();
      np->type = type;
      np->add_input(0, u);
      return np;
    }

    //! get description of an operator for a given type
    const NodeDesc& get_node_desc(NodeType type){

      static bool has_init = false;                                            
      static OpDescList s;
      
      if(!has_init){
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
	 s.insert(s.begin() + ${type}$,{"${name}$", ${ew}$, ${cl}$, "${ef}$"});
#:endfor
	 has_init = true;
      }
      return s.at(type);
    }


    void write_graph(const NodePtr& root){
      
    }
  }
}

#endif
