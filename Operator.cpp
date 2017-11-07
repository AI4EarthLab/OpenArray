
#include "Operator.hpp"
#include "utils/utils.hpp"
#include "Kernel.hpp"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

using namespace oa::kernel;

namespace oa {
  namespace ops{

    NodePtr new_node(const ArrayPtr &ap) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DATA);
      np->set_data(ap);
      np->set_data_type(ap->get_data_type());
      np->set_shape(ap->shape());
      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      int dt = oa::utils::cast_data_type(
        u->get_data_type(),
        v->get_data_type());
      
      const NodeDesc &nd = get_node_desc(type);
      if (nd.ew) {
        np->set_depth(u->get_depth(), v->get_depth());
        // U and V must have same shape
        if (u->is_seqs_scalar()) np->set_shape(v->shape());
        else if (v->is_seqs_scalar()) np->set_shape(u->shape());
        else {
          assert(oa::utils::is_equal_shape(u->shape(), v->shape()));
          np->set_shape(u->shape());
        }
        np->set_data_type(dt);
      } else {
        // to do
        // set data_type && shape
        np->set_data_type(dt);
      }
      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);

      const NodeDesc &nd = get_node_desc(type);
      if (nd.ew) {
        np->set_depth(u->get_depth());
        np->set_shape(u->shape());
        np->set_data_type(u->get_data_type());
      } else {
        // to do set data_type && shape
      }
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
        ///:set sy = i[2]
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
        s[${type}$] = {${type}$, "${name}$", "${sy}$", ${ew}$, ${cl}$, "${ef}$", ${kernel_name}$};
        ///:else
        s[${type}$] = {${type}$, "${name}$", "${sy}$", ${ew}$, ${cl}$, "${ef}$", NULL};
        ///:endif
        ///:set id = id + 1
        ///:endfor
        has_init = true;
      }  
      return s.at(type);
    }

    void write_graph(const NodePtr& root, bool is_root, const char *filename) {
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

    const KernelPtr get_kernel_dict(size_t hash, const char *filename) {
      static bool has_init = false;
      static unordered_map<size_t, KernelPtr> kernel_dict;
      if (!has_init) {
        has_init = true;

        void *handle;
        char *error;

        // open dynamic library
        handle = dlopen(LIB_KERNEL_PATH, RTLD_LAZY);
        if (!handle) {
          fprintf(stderr, "%s\n", error);
          exit(EXIT_FAILURE);
        }

        // clean up the error before
        dlerror();

        // open kernel_dict and get kernel name as function signature
        std::ifstream ifs;
        ifs.open(filename);
        size_t key;
        string value;
        typedef ArrayPtr (*FUNC)(vector<ArrayPtr>&);
        KernelPtr func;

        while(ifs>>key) {
          ifs>>value;
          stringstream ss;
          ss<<"kernel_"<<key;
          func = (FUNC)(dlsym(handle, ss.str().c_str()));
          if ((error = dlerror()) != NULL) {
            fprintf(stderr, "%s\n", error);
            func = NULL;
          }
          kernel_dict[key] = func;
        }

        ifs.close();
        dlclose(handle);
      }
      if (kernel_dict.find(hash) == kernel_dict.end()) return NULL;
      return kernel_dict[hash];
    }

    void insert_kernel_dict(size_t hash, const stringstream &s,
      const char *filename) {
      std::ofstream ofs;
      ofs.open(filename, std::ofstream::out | std::ofstream::app);
      ofs<<hash<<" "<<s.str()<<endl;
      ofs.close();
    }

    void gen_kernels(NodePtr A, bool is_root, MPI_Comm comm) {
      if (oa::utils::get_rank(comm)) return ;
      if (A->has_data()) return ;

      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels(A->input(i), true);
        }
        return ;
      }

      if (is_root && A->get_depth() >= 2) {
        // fusion kernel
        stringstream ss = tree_to_string(A);
        // fusion kernel hash
        stringstream ss1;
        tree_to_string_stack(A, ss1);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss1.str());
        
        const KernelPtr func = get_kernel_dict(hash);
        if (func == NULL) {
          insert_kernel_dict(hash, ss);
        }
      }

      for (int i = 0; i < A->input_size(); i++) {
        gen_kernels(A->input(i), false);
      }
    }

    // example: (A1+S2)*A3
    stringstream tree_to_string(NodePtr A) {
      stringstream ss;
      const NodeDesc &nd = get_node_desc(A->type());
      
      // only data or non-element-wise
      if (A->has_data() || !nd.ew) {
        if (A->is_seqs_scalar()) ss<<"S";
        else ss<<"A";
        ss<<A->get_data_type();
        return ss;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
        case 1:
          ss<<nd.sy<<"("<<child[0].str()<<")";
          break;
        case 2:
          ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
          break;
      }

      return ss;
    }

    void tree_to_string_stack(NodePtr A, stringstream &ss) {
      const NodeDesc &nd = get_node_desc(A->type());

      if (A->has_data() || !nd.ew) {
        if (A->is_seqs_scalar()) ss<<"S";
        else ss<<"A";
        ss<<A->get_data_type();
        return ;
      }

      for (int i = 0; i < A->input_size(); i++) {
        tree_to_string_stack(A->input(i), ss);
      }
      ss<<nd.sy;

      return ;
    }

  }
}


