#include "Operator.hpp"
#include "utils/utils.hpp"
#include "Kernel.hpp"
#include "Jit_Driver.hpp"
#include "Grid.hpp"
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
      np->set_scalar(ap->is_scalar());
      np->set_seqs(ap->is_seqs());
      np->set_pos(ap->get_pos());
      np->set_bitset(ap->get_bitset());
      np->set_pseudo(ap->is_pseudo());

      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      
      const NodeDesc &nd = get_node_desc(type);
      // set dt for ans, ans = u type v
      int dt = nd.rt;
      if (TYPE_PLUS <= type && type <= TYPE_DIVD) {
        dt = oa::utils::cast_data_type(
                                     u->get_data_type(),
                                     v->get_data_type());
      }

      if (nd.ew) {
        np->set_depth(u->get_depth(), v->get_depth());
        // U and V must have same shape
        if (u->is_seqs_scalar()) np->set_shape(v->shape());
        else if (v->is_seqs_scalar()) np->set_shape(u->shape());
        else {
          /*
          pseudo 3d, so don't have to assert
          assert(oa::utils::is_equal_shape(u->shape(), v->shape()));
          */
          np->set_shape(u->shape());
        }
        np->set_data_type(dt);
        np->set_lbound(u->get_lbound(), v->get_lbound());
        np->set_rbound(u->get_rbound(), v->get_rbound());
      } else {
        np->set_lbound({0, 0, 0});
        np->set_rbound({0, 0, 0});
        np->set_update();
        np->set_data_type(dt);
      }
      
      // u & v must in the same grid pos
      //assert(u->get_pos() == v->get_pos());
      if(u->get_pos() != -1)
        np->set_pos(u->get_pos());
      else if(v->get_pos() != -1)
        np->set_pos(v->get_pos());
     
      return np;
    }

    NodePtr new_node(NodeType type, NodePtr u){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      
      const NodeDesc &nd = get_node_desc(type);
      int dt = u->get_data_type();
      if (TYPE_EXP <= type && type <= TYPE_DZF) dt = nd.rt;
      if (type == TYPE_UPLUS or type == TYPE_UMINUS) dt = u->get_data_type();
      if (type == TYPE_NOT) dt = nd.rt;

      // only OP will change grid pos
      np->set_pos(u->get_pos());

      if (nd.ew) {
        np->set_depth(u->get_depth());
        np->set_shape(u->shape());

        np->set_data_type(dt);
        np->set_lbound(u->get_lbound());
        np->set_rbound(u->get_rbound());
        

        if (TYPE_AXB <= type && type <= TYPE_DZF) {
          int3 lb, rb;
          switch (type) {
            case TYPE_AXB:
            case TYPE_DXB:
              lb = {1, 0, 0};
              rb = {0, 0, 0};
              break;
            case TYPE_AXF:
            case TYPE_DXF:
              lb = {0, 0, 0};
              rb = {1, 0, 0}; 
              break;
            case TYPE_AYB:
            case TYPE_DYB:
              lb = {0, 1, 0};
              rb = {0, 0, 0}; 
              break;
            case TYPE_AYF:
            case TYPE_DYF:
              lb = {0, 0, 0};
              rb = {0, 1, 0}; 
              break;
            case TYPE_AZB:
            case TYPE_DZB:
              lb = {0, 0, 1};
              rb = {0, 0, 0}; 
              break;
            case TYPE_AZF:
            case TYPE_DZF:
              lb = {0, 0, 0};
              rb = {0, 0, 1}; 
              break;
          }

          int3 new_lb = u->get_lbound();
          int3 new_rb = u->get_rbound();
          
          int mx = 0;
          for (int i = 0; i < 3; i++) {
            new_lb[i] += lb[i];
            mx = max(new_lb[i], mx);
            new_rb[i] += rb[i];
            mx = max(new_rb[i], mx);
          }

          // set default max stencil as two
          if (mx > 1) {
            np->set_lbound(lb);
            np->set_rbound(rb);
            u->set_update();
          } else {
            np->set_lbound(new_lb);
            np->set_rbound(new_rb);
          }
        }

        if(u->get_pos() != -1){
          np->set_pos(Grid::global()->get_pos(u->get_pos(), type));
        }

      } else {
        // to do set data_type && shape
        np->set_lbound({0, 0, 0});
        np->set_rbound({0, 0, 0});
        np->set_update();
        np->set_data_type(dt);
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
        ///:set rt = i[8]
        ///:set kernel_name = 'kernel_' + i[1]
        

        s[${type}$].type = ${type}$;
        s[${type}$].name = "${name}$";
        s[${type}$].sy = "${sy}$";
        s[${type}$].ew = ${ew}$;
        s[${type}$].cl = ${cl}$;
        s[${type}$].expr = "${ef}$";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        ///:if i[2] == ''
        s[${type}$].func = NULL;
        ///:else
        s[${type}$].func = ${kernel_name}$;
        ///:endif
        
        ///!:else
        ///s[${type}$].func = NULL;
        ///!:endif
        
        s[${type}$].rt = ${rt}$;


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

    ArrayPtr force_eval(NodePtr A) {
      if (A->has_data()) return A->get_data();

      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(force_eval(A->input(i)));
      }

      const NodeDesc& nd = get_node_desc(A->type());
      KernelPtr kernel_addr = nd.func;
      //printf("kernel : %p\n", kernel_addr.target< kernel_rawptr* >());
      ArrayPtr ap = kernel_addr(ops_ap);
      //A->set_data(ap);
      ap->set_pseudo(A->is_pseudo());
      ap->set_bitset(A->get_bitset());
      ap->set_pos(A->get_pos());

      return ap;
    }

    void get_kernel_parameter(NodePtr A, vector<void*> &list, 
      PartitionPtr &ptr) {
      ArrayPtr ap;
      // data
      if (A->has_data()) {
        ap = A->get_data();
        list.push_back(ap->get_buffer());
        if (ptr == NULL && !(ap->is_seqs_scalar())) {
          ptr = ap->get_partition();
        }
        return ;
      }

      // not element wise, need eval
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        ArrayPtr ap = eval(A);
        list.push_back(ap->get_buffer());
        if (ptr == NULL && !(ap->is_seqs_scalar())) {
          ptr = ap->get_partition();
        }
        return ;
      }

      // tree
      for (int i = 0; i < A->input_size(); i++) {
        get_kernel_parameter(A->input(i), list, ptr);
      }
    }

    ArrayPtr eval(NodePtr A) {
      // fusion kernel
      if (A->hash()) {
        FusionKernelPtr fkptr = Jit_Driver::global()->get(A->hash());
        if (fkptr != NULL) {
          vector<void*> list;
          PartitionPtr par_ptr;
          get_kernel_parameter(A, list, par_ptr);
          ArrayPtr ap = ArrayPool::global()->get(par_ptr, A->get_data_type());

          list.push_back(ap->get_buffer());
          void** list_pointer = list.data();
          fkptr(list_pointer, ap->buffer_size());
          // cout<<"fusion-kernel called"<<endl;
          
          //A->set_data(ap);
          ap->set_pseudo(A->is_pseudo());
          ap->set_bitset(A->get_bitset());
          ap->set_pos(A->get_pos());

          return ap;
        }
      }

      
      // data
      if (A->has_data()) return A->get_data();

      // tree
      vector<ArrayPtr> ops_ap;

      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(eval(A->input(i)));
      }

      //printf("ATYPE=%d\n",A->type());
      ArrayPtr ap;
      if(A->type() == TYPE_REF){
        ap = oa::funcs::subarray(ops_ap[0], A->get_ref());
      }else{
        const NodeDesc& nd = get_node_desc(A->type());
        KernelPtr kernel_addr = nd.func;
        ap = kernel_addr(ops_ap);
        //A->set_data(ap);
        ap->set_pseudo(A->is_pseudo());
        ap->set_bitset(A->get_bitset());
        ap->set_pos(A->get_pos());
      }
      // ap->display();

      return ap;
    }


    ArrayPtr eval_JIT(NodePtr A) {
      // data
      if (A->has_data()) return A->get_data();

      // generate hash
      if (!A->hash()) {
        stringstream ss;
        tree_to_string_stack(A, ss);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss.str());
        A->set_hash(hash);
      }

      // fusion kernel
      if (A->hash()) {
        FusionKernelPtr fkptr = Jit_Driver::global()->get(A->hash());
        if (fkptr != NULL) {
          vector<void*> list;
          PartitionPtr par_ptr;
          get_kernel_parameter(A, list, par_ptr);
          ArrayPtr ap = ArrayPool::global()->get(par_ptr, A->get_data_type());

          list.push_back(ap->get_buffer());
          void** list_pointer = list.data();
          fkptr(list_pointer, ap->buffer_size());
          cout<<"fusion-kernel called"<<endl;
          
          //A->set_data(ap);
          ap->set_pseudo(A->is_pseudo());
          ap->set_bitset(A->get_bitset());
          ap->set_pos(A->get_pos());

          return ap;
        }
      }

      // tree
      vector<ArrayPtr> ops_ap;

      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(eval(A->input(i)));
      }

      const NodeDesc& nd = get_node_desc(A->type());
      KernelPtr kernel_addr = nd.func;
      ArrayPtr ap = kernel_addr(ops_ap);
      //A->set_data(ap);
      ap->set_pseudo(A->is_pseudo());
      ap->set_bitset(A->get_bitset());
      ap->set_pos(A->get_pos());


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
      //A->display();
      
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels(A->input(i), true);
        }
        return ;
      }

      if (is_root && A->get_depth() >= 2) {
        // fusion kernel
        //stringstream ss = tree_to_string(A);
        // fusion kernel hash
        stringstream ss;
        stringstream ss1;
        stringstream code;
        //code<<"for (int i = 0; i < size; i++) {\n  ans[i] = ";
        int id = 0;
        tree_to_code(A, code, id);
        cout<<code.str()<<endl;
        tree_to_string(A, ss);
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

    void gen_kernels_JIT(NodePtr A, bool is_root, MPI_Comm comm) {
      //if (oa::utils::get_rank(comm)) return ;
      if (A->has_data()) return ;
      
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels_JIT(A->input(i), true);
        }
        return ;
      }

      if (is_root && A->get_depth() >= 2) {
        stringstream ss1;
        stringstream code;
        stringstream __code;
        //code<<"for (int i = 0; i < size; i++) {\n  ans[i] = ";
        int id = 0;
        vector<int> int_id, float_id, double_id;
        tree_to_code(A, __code, id, int_id, float_id, double_id);
        tree_to_string_stack(A, ss1);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss1.str());
        
        // JIT source code add function signature
        code_add_function_signature(code, hash);
        // JIT source code add const parameters
        code_add_const(code, int_id, float_id, double_id);
        // JIT source code add calc_inside
        code_add_function(code, __code, A->get_data_type(), id);

        cout<<code.str()<<endl;
        // Add fusion kernel into JIT map
        Jit_Driver::global()->insert(hash, code);

        A->set_hash(hash);
      }

      for (int i = 0; i < A->input_size(); i++) {
        gen_kernels_JIT(A->input(i), false);
      }
    }

    void gen_kernels_JIT_with_op(NodePtr A, bool is_root, MPI_Comm comm) {
      //if (oa::utils::get_rank(comm)) return ;
      if (A->has_data()) return ;
      
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels_JIT_with_op(A->input(i), true);
        }
        return ;
      }

      if (is_root && A->get_depth() >= 2) {
        stringstream ss1;
        stringstream code;
        stringstream __code;
        //code<<"for (int i = 0; i < size; i++) {\n  ans[i] = ";
        int id = 0;
        int S_id = 0;
        vector<int> int_id, float_id, double_id;
        tree_to_code_with_op(A, __code, id, S_id, int_id, float_id, double_id);
        tree_to_string_stack(A, ss1);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss1.str());
        
        // JIT source code add function signature
        code_add_function_signature_with_op(code, hash);
        // JIT source code add const parameters
        code_add_const(code, int_id, float_id, double_id);
        // JIT source code add calc_inside
        code_add_calc_inside(code, __code, A->get_data_type(), id, S_id);

        cout<<code.str()<<endl;
        // Add fusion kernel into JIT map
        Jit_Driver::global()->insert(hash, code);

        A->set_hash(hash);
      }

      for (int i = 0; i < A->input_size(); i++) {
        gen_kernels_JIT_with_op(A->input(i), false);
      }
    }

    // example: (A1+S2)*A3
    
    void tree_to_string(NodePtr A, stringstream &ss) {
      const NodeDesc &nd = get_node_desc(A->type());
      
      // only data or non-element-wise
      if (A->has_data() || !nd.ew || A->need_update()) {
        if (A->is_seqs_scalar()) ss<<"S";
        else ss<<"A";
        ss<<A->get_data_type();
        return;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_string(A->input(i), child[i]);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        ss<<nd.sy<<"("<<child[0].str()<<")";
        break;
      case 2:
        ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        break;
      }

      return;
    }

    void tree_to_code(NodePtr A, stringstream &ss, int &id) {
      const NodeDesc &nd = get_node_desc(A->type());

      if (A->has_data() || !nd.ew || A->need_update()) {
        ss<<"(";
        switch(A->get_data_type()) {
          case DATA_INT:
            ss<<"(int*)";
            break;
          case DATA_FLOAT:
            ss<<"(float*)";
            break;
          case DATA_DOUBLE:
            ss<<"(double*)";
            break;
        }
        ss<<"(list["<<id<<"]))";
        if (A->is_seqs_scalar()) ss<<"[0]";
        else ss<<"[i]";
        id++;
        return ;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_code(A->input(i), child[i], id);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        if(nd.sy == "abs")
          ss<<"fabs"<<"("<<child[0].str()<<")";
        else
          ss<<nd.sy<<"("<<child[0].str()<<")";
        break;
      case 2:
        ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        break;
      }

      return;
    }

    void tree_to_code(NodePtr A, stringstream &ss, int &id,
      vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      const NodeDesc &nd = get_node_desc(A->type());

      if (A->has_data() || !nd.ew || A->need_update()) {
        if (A->is_seqs_scalar()) {
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"I_";
              ss<<int_id.size();
              int_id.push_back(id);
              break;
            case DATA_FLOAT:
              ss<<"F_";
              ss<<float_id.size();
              float_id.push_back(id);
              break;
            case DATA_DOUBLE:
              ss<<"D_";
              ss<<double_id.size();
              double_id.push_back(id);
              break;
          }
        } else {
          ss<<"(";
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"(int*)";
              break;
            case DATA_FLOAT:
              ss<<"(float*)";
              break;
            case DATA_DOUBLE:
              ss<<"(double*)";
              break;
          }
          ss<<"(list["<<id<<"]))[i]";
        }
        id++;
        return ;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_code(A->input(i), child[i], id, int_id, float_id, double_id);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        if(nd.sy == "abs")
          ss<<"fabs"<<"("<<child[0].str()<<")";
        else
          ss<<nd.sy<<"("<<child[0].str()<<")";
        break;
      case 2:
        ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        break;
      }

      return;
    }

    void tree_to_code_with_op(NodePtr A, stringstream &ss, int &id, int& S_id,
      vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      const NodeDesc &nd = get_node_desc(A->type());

      // data node
      if (A->has_data() || !nd.ew || A->need_update()) {
        // scalar
        if (A->is_seqs_scalar()) {
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"I_";
              ss<<int_id.size();
              int_id.push_back(id);
              break;
            case DATA_FLOAT:
              ss<<"F_";
              ss<<float_id.size();
              float_id.push_back(id);
              break;
            case DATA_DOUBLE:
              ss<<"D_";
              ss<<double_id.size();
              double_id.push_back(id);
              break;
          }
        // [i][j][k] based on node bitset
        } else {
          ss<<"(";
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"(int*)";
              break;
            case DATA_FLOAT:
              ss<<"(float*)";
              break;
            case DATA_DOUBLE:
              ss<<"(double*)";
              break;
          }
          ss<<"(list["<<id<<"]))";
          
          char pos_i[3] = "oi";
          char pos_j[3] = "oj";
          char pos_k[3] = "ok";
          
          bitset<3> bit = A->get_bitset();
          ss<<"[calc_id("<<pos_i[bit[2]]<<",";
          ss<<pos_j[bit[1]]<<",";
          ss<<pos_k[bit[0]]<<",S"<<S_id<<")]";
          S_id++;
        }
        id++;
        return ;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_code_with_op(A->input(i), child[i], id, S_id, 
          int_id, float_id, double_id);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        change_string_with_op(ss, child[0].str(), nd);
        // bind grid if A.pos != -1
        if (A->get_pos() != -1) {
          if (nd.type == TYPE_DXC ||
            nd.type == TYPE_DYC ||
            nd.type == TYPE_DZC ||
            nd.type == TYPE_DXB ||
            nd.type == TYPE_DXF ||
            nd.type == TYPE_DYB ||
            nd.type == TYPE_DYF ||
            nd.type == TYPE_DZB ||
            nd.type == TYPE_DZF) {

            // get grid ptr
            ArrayPtr grid_ptr = Grid::global()->get_grid(A->get_pos(), nd.type);

            ss<<"/(";
            switch(grid_ptr->get_data_type()) {
              case DATA_INT:
                ss<<"(int*)";
                break;
              case DATA_FLOAT:
                ss<<"(float*)";
                break;
              case DATA_DOUBLE:
                ss<<"(double*)";
                break;
            }
            ss<<"(list["<<id<<"]))";
            id++;
            
            char pos_i[3] = "oi";
            char pos_j[3] = "oj";
            char pos_k[3] = "ok";
            
            bitset<3> bit = grid_ptr->get_bitset();
            ss<<"[calc_id("<<pos_i[bit[2]]<<",";
            ss<<pos_j[bit[1]]<<",";
            ss<<pos_k[bit[0]]<<",S"<<S_id<<")]";
            S_id++;
          }
        }
        break;
      case 2:
        ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        break;
      }

      return;
    }

    void change_string_with_op(stringstream& ss, string in, const NodeDesc &nd) {
      string new_str1, new_str2, new_str;
      switch(nd.type) {
      // Central difference operator
      case TYPE_DXC:
        new_str1 = replace_string(in, "i,", "+1+i,");
        new_str2 = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*("<<new_str1<<"-"<<new_str2<<")";
        break;
      case TYPE_DYC:
        new_str1 = replace_string(in, "j,", "+1+j,");
        new_str2 = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*("<<new_str1<<"-"<<new_str2<<")";
        break;
      case TYPE_DZC:
        new_str1 = replace_string(in, "k,", "+1+k,");
        new_str2 = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*("<<new_str1<<"-"<<new_str2<<")";
        break;

      // average operator
      case TYPE_AXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;
      case TYPE_AXF:
        new_str = replace_string(in, "i,", "+1+i,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;
      case TYPE_AYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;
      case TYPE_AYF:
        new_str = replace_string(in, "j,", "+1+j,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;
      case TYPE_AZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;
      case TYPE_AZF:
        new_str = replace_string(in, "k,", "+1+k,");
        ss<<"0.5*("<<in<<"+"<<new_str<<")";
        break;

      // difference operator
      case TYPE_DXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"1.0*("<<in<<"-"<<new_str<<")";
        break;
      case TYPE_DXF:
        new_str = replace_string(in, "i,", "+1+i,");
        ss<<"1.0*("<<new_str<<"-"<<in<<")";
        break;
      case TYPE_DYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"1.0*("<<in<<"-"<<new_str<<")";
        break;
      case TYPE_DYF:
        new_str = replace_string(in, "j,", "+1+j,");
        ss<<"1.0*("<<new_str<<"-"<<in<<")";
        break;
      case TYPE_DZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"1.0*("<<in<<"-"<<new_str<<")";
        break;
      case TYPE_DZF:
        new_str = replace_string(in, "k,", "+1+k,");
        ss<<"1.0*("<<new_str<<"-"<<in<<")";
        break;

      // abs operator
      case TYPE_ABS:
        ss<<"fabs"<<"("<<in<<")";
        break;

      // other default monocular operator
      default:
        ss<<nd.sy<<"("<<in<<")";
        break;
      }
    }
    
    // replace all old_str in string in by new_str
    string replace_string(string& in, const string& old_str, const string& new_str) {
      string out = in;
      // use replace is not efficient, should be optimized later
      for(string::size_type i = 0; (i = out.find(old_str, i)) != string::npos;) {
        out.replace(i, old_str.length(), new_str);
        i += new_str.length();
      }
      return out;
    }



    void tree_to_string_stack(NodePtr A, stringstream &ss) {
      const NodeDesc &nd = get_node_desc(A->type());

      if (A->has_data() || !nd.ew || A->need_update()) {
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

    void code_add_function_signature(stringstream& code, size_t& hash) {
      code<<"extern \"C\" {\nvoid kernel_"<<hash;
      code<<"(void** &list, int size) {\n";
    }

    void code_add_function_signature_with_op(stringstream& code, size_t& hash) {
      code<<"extern \"C\" {\nvoid kernel_"<<hash;
      code<<"(void** &list, ) {\n";
    }

    void code_add_const(stringstream& code, 
        vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      for (int i = 0; i < int_id.size(); i++) {
        code<<"const int I_"<<i<<" = ((int*)list["<<int_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < float_id.size(); i++) {
        code<<"const float F_"<<i<<" = ((float*)list["<<float_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < double_id.size(); i++) {
        code<<"const double D_"<<i<<" = ((double*)list["<<double_id[i]<<"])[0];\n";
      }
      code<<"\n";
    }
    
    void code_add_function(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id) {

      code<<"  for (int i = 0; i < size; i++) {\n";  
      switch(dt) {
        case DATA_INT:
          code<<"    ((int*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_FLOAT:
          code<<"    ((float*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_DOUBLE:
          code<<"    ((double*)(list["<<id<<"]))[i] = ";
          break;    
      }
      code<<__code.str()<<";\n  }\n  return ;\n}}";
    }

    void code_add_calc_outside(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id, int& S_id) {
      
      code<<"  for (int i = 0; i < size; i++) {\n";  
      switch(dt) {
        case DATA_INT:
          code<<"    ((int*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_FLOAT:
          code<<"    ((float*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_DOUBLE:
          code<<"    ((double*)(list["<<id<<"]))[i] = ";
          break;    
      }
      code<<__code.str()<<";\n  }\n  return ;\n}}";
    }

    void code_add_calc_inside(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id, int& S_id) {
      
      code<<"  for (int k = o + lbound[2]; k < o + sp[2] - rbound[2]; k++) {\n";
      code<<"    for (int j = o + lbound[1]; j < o + sp[1] - rbound[1]; j++) {\n";
      code<<"      for (int i = o + lbound[0]; i < o + sp[0] - rbound[0]; i++) {\n";

      switch(dt) {
        case DATA_INT:
          code<<"        ((int*)(list["<<id<<"]))[calc_id(i,j,k,S"<<S_id<<")] = ";
          break;
        case DATA_FLOAT:
          code<<"        ((float*)(list["<<id<<"]))[calc_id(i,j,k,S"<<S_id<<")] = ";
          break;
        case DATA_DOUBLE:
          code<<"        ((double*)(list["<<id<<"]))[calc_id(i,j,k,S"<<S_id<<")] = ";
          break;    
      }

      code<<__code.str()<<";\n      }\n    }\n  }\n  return ;\n}}";

    }
    

  }
}
