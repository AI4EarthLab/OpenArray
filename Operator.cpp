
#include "Operator.hpp"
#include "utils/utils.hpp"
#include "Kernel.hpp"
#include <fstream>

using namespace oa::kernel;

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
      
      if (!has_init) {
        s.resize(NUM_NODE_TYPES);
	///:mute
	///:set i = 0  
	///:include "NodeType.fypp"
	///:endmute
	//intialize node descriptions.
	///:set id = 0
	///:for i in L
	///:mute
	///:set type = i[0]
	///:set name = i[1]
	///:set ew = i[5]
	///:if ew == 'F'
	///:set ew = 'false'
	///:else
	///:set ew = 'true'
	///:endif
	///:set cl = i[6]
	///:if cl == 'F'
	///:set cl = 'false'
	///:else
	///:set cl = 'true'
	///:endif
	///:endmute
	///:set ef = i[7]
	///:set kernel_name = 'kernel_' + i[1]
	///:if (i[3] == 'A')
        s[${type}$] = {${type}$, "${name}$", ${ew}$, ${cl}$, "${ef}$", ${kernel_name}$};
	///:else
        s[${type}$] = {${type}$, "${name}$", ${ew}$, ${cl}$, "${ef}$", NULL};
	///:endif
	///:set id = id + 1
	///:endfor
        has_init = true;
      }
      return s.at(type);
    }

    void write_graph(const NodePtr& root, bool is_root, char const *filename) {
      if (oa::utils::get_rank() > 0) return ;
      static std::ofstream ofs;
      if (is_root) {
        ofs.open(filename);
        ofs<<"digraph G {"<<endl;
      }
      int id = root->get_id();
      ofs<<id;

      const NodeDesc & nd = get_node_desc(root->type());
      ofs<<boost::format("[label=\"[%s]\\n id=%d\"];") % nd.name % id<<endl;

      for (int i = 0; i < root->input_size(); i++) {
        write_graph(root->input(i), false, filename);
        ofs<<id<<"->"<<root->input(i)->get_id()<<";"<<endl;
      }

      if (is_root) {
        ofs<<"}"<<endl;
        ofs.close();
      }
    }

    ArrayPtr eval(NodePtr A) {
      if (A->has_data()) return A->get_data();

      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(eval(A->input(i)));
      }

      const NodeDesc& nd = get_node_desc(A->type());
      KernelPtr kernel_addr = nd.func;
      //printf("kernel : %p\n", kernel_addr.target< kernel_rawptr* >());
      ArrayPtr ap = kernel_addr(ops_ap);
      A->set_data(ap);

      return ap;
    }

  }
}


