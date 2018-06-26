/*
 * Operator.cpp
 * evaluate the expression graph
 *
=======================================================*/

#include "Operator.hpp"
#include "utils/utils.hpp"
#include "utils/calcTime.hpp"
#include "Kernel.hpp"
#include "Jit_Driver.hpp"
#include "Grid.hpp"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "Diagnosis.hpp"
#include <boost/format.hpp>

using namespace oa::kernel;

namespace oa {
  namespace ops{

    // needs to set all attributes to the new node
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
      np->set_data_list_size(1);

      return np;
    }

    // only operator min_max & rep will call this function
    // other binary operator will call new_node_type in modules
    NodePtr new_node(NodeType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      np->set_lbound({{0, 0, 0}});
      np->set_rbound({{0, 0, 0}});
      np->set_update();
      np->set_data_type(u->get_data_type());
      
      if(u->get_pos() != -1)
        np->set_pos(u->get_pos());
      else if(v->get_pos() != -1)
        np->set_pos(v->get_pos());
     
      return np;
    }

    const NodeDesc& get_node_desc(NodeType type){

      static bool has_init = false;                                            
      static OpDescList s;
      
      // use NodeType.fypp to initialize the NodeDesc
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
        ///:if (i[1] == 'unknown' or i[2] == '')
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

    // write the expression graph into filename.dot
    void write_graph(const NodePtr& root, bool is_root,
            const char *filename) {
      if (MPI_RANK > 0) return ;
      static std::ofstream ofs;
      if (is_root) {
        ofs.open(filename);
        ofs<<"digraph G {"<<endl;
      }
      int id = root->get_id();
      ofs<<id;

      const NodeDesc & nd = get_node_desc(root->type());

      //char buffer[500];

      //sprintf(buffer, "[label=\"[%s]\\n id=%d \n (ref:%d) "
			// "\n (lb:%d %d %d) \n (rb: %d %d %d) \n (pseudo: %d) \n (up: %d)\"];",  
	    //nd.name, id, root.use_count(),
	    //root->get_lbound()[0], root->get_lbound()[1], root->get_lbound()[2],
	    //root->get_rbound()[0], root->get_rbound()[1], root->get_rbound()[2],
	    //root->is_pseudo(), root->need_update());

      //ofs<<buffer<<endl;

      ofs<<boost::format("[label=\"[%s]\\n id=%d \n (ref:%d) "
                          "\n (lb:%d %d %d) \n (rb: %d %d %d) \n (up: %d)\"];") 
        % nd.name % id % root.use_count()
        % root->get_lbound()[0] % root->get_lbound()[1] % root->get_lbound()[2]
        % root->get_rbound()[0] % root->get_rbound()[1] % root->get_rbound()[2] 
        % root->need_update() <<endl;

      for (int i = 0; i < root->input_size(); i++) {
        write_graph(root->input(i), false, filename);
        ofs<<id<<"->"<<root->input(i)->get_id()<<";"<<endl;
      }

      if (is_root) {
        ofs<<"}"<<endl;
        ofs.close();
      }
    }

    // force eval the expression graph, only use basic kernels
    ArrayPtr force_eval(NodePtr A) {
      if (A->has_data()) return A->get_data();

      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(force_eval(A->input(i)));
      }

      const NodeDesc& nd = get_node_desc(A->type());
      KernelPtr kernel_addr = nd.func;
      ArrayPtr ap = kernel_addr(ops_ap);
      ap->set_pseudo(A->is_pseudo());
      ap->set_bitset(A->get_bitset());
      ap->set_pos(A->get_pos());

      return ap;
    }

    // based on specific NodeType to change the lbound
    int3 change_lbound(NodeType type, int3 lb) {
      switch (type) {
        case TYPE_AXB:
        case TYPE_DXB:
        case TYPE_DXC:
          lb[0] = 1;
          break;
        case TYPE_AYB:
        case TYPE_DYB:
        case TYPE_DYC:
          lb[1] = 1;
          break;
        case TYPE_AZB:
        case TYPE_DZB:
        case TYPE_DZC:
          lb[2] = 1;
          break;
        default:
          break;
      }
      return lb;
    }

    // based on specific NodeType to change the rbound
    int3 change_rbound(NodeType type, int3 rb) {
      switch (type) {
        case TYPE_AXF:
        case TYPE_DXF:
        case TYPE_DXC:
          rb[0] = 1;
          break;
        case TYPE_AYF:
        case TYPE_DYF:
        case TYPE_DYC:
          rb[1] = 1;
          break;
        case TYPE_AZF:
        case TYPE_DZF:
        case TYPE_DZC:
          rb[2] = 1;
          break;
        default:
          break;
      }
      return rb;
    }

    //
    // =======================================================
    // to evaluate expression like A = B + C + D
    // we need to pass parameters to the fusion kernel
    // like: the data pointer to ans A, parameters B, C & D
    //       the shape of ans A, parameters B, C & D etc.
    // =======================================================
    //
    // NodePtr A :  the root of (sub)expression graph
    // list:        the data list which used in fusion kernel
    // update_list: the array list which needs to update boundary 
    // S:           the shape of array in data list which used in fusion kernel
    // ptr:         the final Partition pointer of ans
    // bt:          the final bitset of ans
    // lb_list:     the lbound list of array in data list which used in fusion kernel
    // rb_list:     the rbound list of array in data list which used in fusion kernel
    // lb_now:      the lbound from the root to the current node
    // rb_now:      the rbound from the root to the current node
    // data_list:   the data list of different shape, to check whether data has to transfer or not
    //
    void get_kernel_parameter_with_op(NodePtr A, vector<void*> &list, 
      vector<ArrayPtr> &update_list, vector<int3> &S, PartitionPtr &ptr, 
      bitset<3> &bt, vector<int3> &lb_list, vector<int3> &rb_list,
      int3 lb_now, int3 rb_now, vector<ArrayPtr> &data_list) {
      bool find_in_data_list;
      ArrayPtr ap;
      // 1. the Node is a data node, put data into list
      if (A->has_data()) {
        ap = A->get_data();

        // 1.1 to check whether the ap needs to transfer or not
        if(!ap->is_scalar())
        {
          find_in_data_list = false;
          for(int i = 0; i < data_list.size(); i++){
            if(ap->shape() == data_list[i]->shape()){
              PartitionPtr pp = ap->get_partition();
              if(!(pp->equal(data_list[i]->get_partition()))){
                // ap has the same shape with data_list[i], but the partition is not the same
                ap = oa::funcs::transfer(ap, data_list[i]->get_partition());
              }
              find_in_data_list  = true;
              break;
            }
          }
          // it's the first time the shape appears
          if(!find_in_data_list) data_list.push_back(ap);
        }

        // 1.2 ap is a pseudo 3d, need to make_pseudo_3d
        if (ap->get_bitset() != bt && !ap->is_seqs_scalar() && ap->is_pseudo()) {
          if (ap->has_pseudo_3d() == false) {
            ap->set_pseudo_3d(oa::funcs::make_psudo3d(ap));
          }
          ap = ap->get_pseudo_3d();
        }

        // 1.3 put the array's data into list
        list.push_back(ap->get_buffer());

        // 1.4 determine the answer's partition
        if (ptr == NULL && ap->get_bitset() == bt) {
          ptr = ap->get_partition();
        }

        // 1.5 put the buffer shape into S which needs in fusion kernel
        // put the lb_now & rb_now into lb_list & rb_list which needs in update boundary
        if (!A->is_seqs_scalar()) {
          S.push_back(ap->buffer_shape());
          update_list.push_back(ap);
          lb_list.push_back(lb_now);
          rb_list.push_back(rb_now);
        }
        return ;
      }

      // 2. Operator node is not element wise, or need update 
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        // need change need_update's state in order to evaluate recursively
        bool flag = A->need_update();
        A->set_update(false);
        ArrayPtr ap = eval(A);
        A->set_update(flag);

        // 2.1 to check whether the ap needs to transfer or not
        if(!ap->is_scalar())
        {
          find_in_data_list = false;
          for(int i = 0; i < data_list.size(); i++){
            if(ap->shape() == data_list[i]->shape()){
              PartitionPtr pp = ap->get_partition();
              if(!(pp->equal(data_list[i]->get_partition()))){
                ap = oa::funcs::transfer(ap, data_list[i]->get_partition());
              }
              find_in_data_list  = true;
              break;
            }
          }
          if(!find_in_data_list) data_list.push_back(ap);
        }

        // 2.2 ap is a pseudo 3d, need to make_pseudo_3d
        if (ap->get_bitset() != bt && !ap->is_seqs_scalar() && ap->is_pseudo()) {
          if (ap->has_pseudo_3d() == false) {
            ap->set_pseudo_3d(oa::funcs::make_psudo3d(ap));
          }
          ap = ap->get_pseudo_3d();
        }

        // 2.3 put the array's data into list
        list.push_back(ap->get_buffer());

        // 2.4 determine the answer's partition
        if (ptr == NULL && ap->get_bitset() == bt) {
          ptr = ap->get_partition();
        }

        // 2.5 put the buffer shape into S which needs in fusion kernel
        // put the lb_now & rb_now into lb_list & rb_list which needs in update boundary
        if (!A->is_seqs_scalar()) {
          S.push_back(ap->buffer_shape());
          update_list.push_back(ap);
          lb_list.push_back(lb_now);
          rb_list.push_back(rb_now);
        }
        return ;
      }

      // 3. it's an operator node, get kernel parameters from it's child node 
      for (int i = 0; i < A->input_size(); i++) {
        get_kernel_parameter_with_op(A->input(i), list, update_list, S, ptr, bt,
            lb_list, rb_list, change_lbound(nd.type, lb_now), change_rbound(nd.type, rb_now), data_list);
      }

      // 4. if A is OPERATOR, need to bind grid if A.pos != -1
      if (A->input_size() == 1 && A->get_pos() != -1) {
        if (nd.type == TYPE_DXC ||
            nd.type == TYPE_DYC ||
            nd.type == TYPE_DZC ||
            nd.type == TYPE_DXB ||
            nd.type == TYPE_DXF ||
            nd.type == TYPE_DYB ||
            nd.type == TYPE_DYF ||
            nd.type == TYPE_DZB ||
            nd.type == TYPE_DZF) {

          // 4.1 get grid ptr
          ArrayPtr grid_ptr = Grid::global()->get_grid(A->get_pos(), nd.type);          
          // 4.2 get the grid's data into list
          list.push_back(grid_ptr->get_buffer());
          // 4.3 put the buffer shape into S
          S.push_back(grid_ptr->buffer_shape());
          //if (g_debug) grid_ptr->display("test grid");
        }
      }
    }

    // =======================================================
    // evaluate the expression graph, which the root node is A
    // treat operator as element wise
    //
    //    case 1: if A has fusion kernel, use it to evaluate and return
    //    case 2: if A is a data node, just return it's data 
    //    case 3: if A is not an element wise operator node 
    //            or need to update, evaluate it's child first,
    //            after that, evaluate the A
    // =======================================================
    ArrayPtr eval(NodePtr A) {
      // 1. Node has hash value, means may have a fusion kernel
      if (A->hash()) {
        // use A->hash() to get inside fusion kernel
        FusionKernelPtr fkptr = Jit_Driver::global()->get(A->hash());
        if (fkptr != NULL) {
          // prepare parameters used in fusion kernel
          vector<void*> list;
          vector<int3> S;
          vector<ArrayPtr> update_list;
          PartitionPtr par_ptr;
          bitset<3> bt = A->get_bitset();
          vector<int3> lb_list;
          vector<int3> rb_list;
          int3 lb_now = {{0,0,0}};
          int3 rb_now = {{0,0,0}};
          vector<ArrayPtr> data_list;
          get_kernel_parameter_with_op(A, 
            list, update_list, S, par_ptr, bt,
            lb_list, rb_list, lb_now, rb_now,data_list);

          int3 lb = A->get_lbound();
          int3 rb = A->get_rbound();
          
          int sb = lb[0] + lb[1] + lb[2] + rb[0] + rb[1] + rb[2];
          int sz = update_list.size();
          vector<MPI_Request>  reqs_list;
          // pthread_t tid;
          // step 1:  start of update boundary
          if (sb) {
            for (int i = 0; i < sz; i++){
              oa::funcs::update_ghost_start(update_list[i], reqs_list, 4, lb_list[i], rb_list[i]);
            }
            oa::funcs::update_ghost_end(reqs_list);
          }

          // put the answer array's data and shape into list
          ArrayPtr ap = ArrayPool::global()->get(par_ptr, A->get_data_type());
          S.push_back(ap->buffer_shape());
          S.push_back(A->get_lbound());
          S.push_back(A->get_rbound());
          S.push_back(ap->local_shape());

          list.push_back(ap->get_buffer());
          list.push_back((void*)S.data());
          void** list_pointer = list.data();
          
          // step 2:  calc_inside
          fkptr(list_pointer, ap->get_stencil_width());
          
          if (sb) {
            // step 3:  end of update boundary
              //oa::funcs::update_ghost_end(reqs_list);
            //oa::MPI::wait_end(&tid);

            // step 4:  calc_outside
            // use A->hash() + 1 to get outside fusion kernel
            //FusionKernelPtr out_fkptr = Jit_Driver::global()->get(A->hash() + 1);
            //if (out_fkptr) out_fkptr(list_pointer, ap->get_stencil_width());

            // set the boundary to zeros based on lb & rb becased it used illegal data
            //oa::funcs::set_boundary_zeros(ap, lb, rb);
          }
            oa::funcs::set_boundary_zeros(ap, lb, rb);

          //cout<<"fusion-kernel called"<<endl;
          
          ap->set_bitset(A->get_bitset());
          ap->set_pos(A->get_pos());
          return ap;
        }
      }

      
      // 2. Node is a data node, just return the data
      if (A->has_data()) return A->get_data();

      // 3.1 Node is an operator node, and doesn't have fusion kernel
      // first, evaluate it's child node recursively
      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(eval(A->input(i)));
      }

      // 3.2 second, evaluate the node
      ArrayPtr ap;
      if(A->type() == TYPE_REF) {
        ap = oa::funcs::subarray(ops_ap[0], A->get_ref());
      } else {
        const NodeDesc& nd = get_node_desc(A->type());
        KernelPtr kernel_addr = nd.func;

        ap = kernel_addr(ops_ap);
        ap->set_bitset(A->get_bitset());
        ap->set_pos(A->get_pos());
      }
      return ap;
    }


    // =======================================================
    // Before evaluate the expression graph, we need to analyze the graph
    // Here's how we generate the fusion kernel for each sub expression graph
    // 
    // exp: A = AXF(A) + sub(A+B+C) 
    // there is two fusion kernels becasue of the sub operator
    //      kernel 1: ans = AXF(A) + tmp
    //      kernel 2: ans = A+B+C
    // =======================================================
    // NodePtr A :  the root of (sub)expression graph
    // is_root:     we only have to generate fusion kernels of the root node
    void gen_kernels_JIT_with_op(NodePtr A, bool is_root) {
      // 1. if A is a data node, doesn't have to generate fusion kernel
      if (A->has_data()) return ;
      
      const NodeDesc &nd = get_node_desc(A->type());
      
      // 2. if A is not element wise (like sum, rep, etc)
      //    need to generate it's children's fusion kernels recursively
      if (!nd.ew) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels_JIT_with_op(A->input(i), true);
        }
        return ;
      }

      // 3. if A's need update state is true, should generate fusion kernel 
      if (A->need_update()) {
        // should set update to false in order to generate kernels
        A->set_update(false);
        gen_kernels_JIT_with_op(A, true);
        A->set_update(true);
        return ;
      }

      // 4. is root && A->depth >= 2, generate fusion kernel
      if (is_root && A->get_depth() >= 2) {
        stringstream ss1;
        stringstream code;
        stringstream __code;
        stringstream __point;
        
        // generate hash code for tree
        tree_to_string_stack(A, ss1);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss1.str());
        if (g_debug) cout<<ss1.str()<<endl;
        if (g_debug) cout<<hash<<endl;
        
        // if already have kernel function ptr, do nothing
        if (Jit_Driver::global()->get(hash) != NULL) {
          if (g_debug) cout<<hash<<endl;
          A->set_hash(hash);
          // return ;   shouldn't return!!!!
        } 
        else {
          // else generate kernel function by JIT_Driver
          int id = 0;
          int S_id = 0;
          vector<int> int_id, float_id, double_id;
          tree_to_code_with_op(A, __code, __point, id, S_id, int_id, float_id, double_id);
          
          // JIT source code add function signature
          code_add_function_signature_with_op(code, hash);
          // JIT source code add const parameters
          code_add_const(code, int_id, float_id, double_id);
          // JIT source code add calc_inside
          code_add_calc_inside(code, __code, __point, A->get_data_type(), id, S_id);

          // for debug
          if (g_debug) cout<<code.str()<<endl;

          // Add fusion kernel into JIT map
          Jit_Driver::global()->insert(hash, code);

          A->set_hash(hash);

          int3 lb = A->get_lbound();
          int3 rb = A->get_rbound();
          int sb = lb[0] + lb[1] + lb[2] + rb[0] + rb[1] + rb[2];

          // Add calc_outside
          if (sb) {
            //stringstream code_out;
            //size_t hash_out = hash + 1;
            //code_add_function_signature_with_op(code_out, hash_out);
            //code_add_const(code_out, int_id, float_id, double_id);
            //code_add_calc_outside(code_out, __code, A->get_data_type(), id, S_id);
            //// cout<<code_out.str()<<endl;
            //Jit_Driver::global()->insert(hash_out, code_out);
          }
        }
      }

      // 5. generate fusion kernels recursively 
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
        if(nd.sy == "abs")
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"abs"<<"("<<child[0].str()<<")";
              break;
            case DATA_FLOAT:
              ss<<"fabsf"<<"("<<child[0].str()<<")";
              break;
            case DATA_DOUBLE:
              ss<<"fabs"<<"("<<child[0].str()<<")";
              break;    
            default:
              ss<<"fabs"<<"("<<child[0].str()<<")";
              break;    
          }
        else if(nd.sy == "sqrt")
          switch(A->get_data_type()) {
            case DATA_FLOAT:
              ss<<"sqrtf"<<"("<<child[0].str()<<")";
              break;
            default:
              ss<<"sqrt"<<"("<<child[0].str()<<")";
              break;    
          }
        else
          ss<<nd.sy<<"("<<child[0].str()<<")";
        break;
      case 2:
        if (nd.type == TYPE_POW) {
          ss<<"pow("<<child[0].str()<<","<<child[1].str()<<")";
        }
        else {
          ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        }
        break;
      }

      return;
    }

    void tree_to_code_with_op(NodePtr A, stringstream &ss, stringstream &__point, int &id, int& S_id,
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
          switch(A->get_data_type()) {
            case DATA_INT:
                __point<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";
              break;
            case DATA_FLOAT:
                __point<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";
              break;
            case DATA_DOUBLE:
                __point<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";
              break;    
          }
        /*
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
          */
          ss<<"list_"<<id;
          
          char pos_i[3] = "oi";
          char pos_j[3] = "oj";
          char pos_k[3] = "ok";
          
          bitset<3> bit = A->get_bitset();
          ss<<"[calc_id2("<<pos_i[bit[2]]<<",";
          ss<<pos_j[bit[1]]<<",";
          ss<<pos_k[bit[0]]<<",S"<<S_id<<"_0,S"<<S_id<<"_1)]";
          S_id++;
        }
        id++;
        return ;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_code_with_op(A->input(i), child[i], __point, id, S_id, 
          int_id, float_id, double_id);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        if (nd.type == TYPE_UNKNOWN) {
          string in = child[0].str();
          // printf("in Operator, k = %d\n", A->get_slice());
          string out  = replace_string(in, "k", to_string(A->get_slice()));
          ss<<out;
        }
        else change_string_with_op(ss, child[0].str(), nd);
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

            ss<<"/";
            switch(A->get_data_type()) {
              case DATA_INT:
                __point<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";
                break;
              case DATA_FLOAT:
                __point<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";
                break;
              case DATA_DOUBLE:
                __point<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";
                break;    
            }
            /*
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
            }*/
            ss<<"list_"<<id;
            id++;
            
            char pos_i[3] = "oi";
            char pos_j[3] = "oj";
            char pos_k[3] = "ok";
            bitset<3> bit = grid_ptr->get_bitset();
            ss<<"[calc_id2("<<pos_i[bit[2]]<<",";
            ss<<pos_j[bit[1]]<<",";
            ss<<pos_k[bit[0]]<<",S"<<S_id<<"_0,S"<<S_id<<"_1)]";
            S_id++;
          }
        }
        break;
      case 2:
        if (nd.type == TYPE_POW) {
          ss<<"pow("<<child[0].str()<<","<<child[1].str()<<")";
        }
        else {
          ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        }
        break;
      }

      return;
    }

    void change_string_with_op(stringstream& ss, string in, const NodeDesc &nd) {
      string new_str1, new_str2, new_str;
      switch(nd.type) {
      // Central difference operator
      case TYPE_DXC:
        new_str1 = replace_string(in, "i,", "1+i,");
        new_str2 = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;
      case TYPE_DYC:
        new_str1 = replace_string(in, "j,", "1+j,");
        new_str2 = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;
      case TYPE_DZC:
        new_str1 = replace_string(in, "k,", "1+k,");
        new_str2 = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;

      // average operator
      case TYPE_AXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AXF:
        new_str = replace_string(in, "i,", "1+i,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AYF:
        new_str = replace_string(in, "j,", "1+j,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AZF:
        new_str = replace_string(in, "k,", "1+k,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;

      // difference operator
      case TYPE_DXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DXF:
        new_str = replace_string(in, "i,", "1+i,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
        break;
      case TYPE_DYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DYF:
        new_str = replace_string(in, "j,", "1+j,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
        break;
      case TYPE_DZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DZF:
        new_str = replace_string(in, "k,", "1+k,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
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
        else {
          ss<<"A"<<A->get_bitset();
        }
        ss<<A->get_data_type();

        // if (A->need_update()) ss<<nd.sy;
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

      // code<<"#include <array>\n\n";
      // code<<"typedef std::array<int, 3> int3;\n\n";
      code<<"#include \"math.h\"\n";
      code<<"#include \"stdlib.h\"\n";
      code<<"#include \"stdio.h\"\n";
      
      code<<"typedef int int3[3];\n\n";
      code<<"#define min(a,b) ((a)<(b))?(a):(b)\n";
      code<<"#define BLOCK_NUM 32\n";

      code<<"extern \"C\" {\n";
      code<<"#define calc_id2(i,j,k,S0,S1) ((k)*(S0)*(S1)+(j)*(S0)+(i))\n";
      code<<"void kernel_"<<hash;
      code<<"(void** &list, int o) {\n";
    }

    void code_add_const(stringstream& code, 
        vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      code<<"\n";
      for (int i = 0; i < int_id.size(); i++) {
        code<<"  const int I_"<<i<<" = ((int*)list["<<int_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < float_id.size(); i++) {
        code<<"  const float F_"<<i<<" = ((float*)list["<<float_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < double_id.size(); i++) {
        code<<"  const double D_"<<i<<" = ((double*)list["<<double_id[i]<<"])[0];\n";
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
/*
    void code_add_calc_outside(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id, int& S_id) {
      
      code<<"  int3* int3_p = (int3*)(list["<<id + 1<<"]);\n";
      for (int i = 0; i <= S_id; i++) {
        code<<"  const int3 &S"<<i<<" = int3_p["<<i<<"];\n";
        code<<"  const int S"<<i<<"_0 = int3_p["<<i<<"][0];\n";
        code<<"  const int S"<<i<<"_1 = int3_p["<<i<<"][1];\n";
      }
      code<<"\n";
      code<<"  const int3 &lbound = int3_p["<<S_id + 1<<"];\n";
      code<<"  const int3 &rbound = int3_p["<<S_id + 2<<"];\n";
      code<<"  const int3 &sp = int3_p["<<S_id + 3<<"];\n\n";

      string ans_type[3];
      ans_type[DATA_INT] = "(int*)";
      ans_type[DATA_FLOAT] = "(float*)";
      ans_type[DATA_DOUBLE] = "(double*)";

      // lbound[2]
      code<<"  if (lbound[2]) {\n";
      code<<"    for (int k = o; k < o + lbound[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[2]
      code<<"  if (rbound[2]) {\n";
      code<<"    for (int k = o + sp[2] - rbound[2]; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";


      // lbound[1]
      code<<"  if (lbound[1]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + lbound[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[1]
      code<<"  if (rbound[1]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o + sp[1] - rbound[1]; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";

      // lbound[0]
      code<<"  if (lbound[0]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"        for (int i = o; i < o + lbound[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[0]
      code<<"  if (rbound[0]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"        for (int i = o + sp[0] - rbound[0]; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";

      code<<"  return ;\n}}";

    }
    */

    void code_add_calc_inside(stringstream& code, 
      stringstream& __code, stringstream& __point, DATA_TYPE dt, int& id, int& S_id) {
      code<<"  //o = 1;//temp wangdong\n";
      code<<"  int3* int3_p = (int3*)(list["<<id + 1<<"]);\n";
      for (int i = 0; i <= S_id; i++) {
        //code<<"  const int3 &S"<<i<<" = int3_p["<<i<<"];\n";
        code<<"  const int S"<<i<<"_0 = int3_p["<<i<<"][0];  ";
        code<<"  const int S"<<i<<"_1 = int3_p["<<i<<"][1];\n";
      }
      code<<"\n";
      code<<"  const int3 &lbound = int3_p["<<S_id + 1<<"];\n";
      code<<"  const int3 &rbound = int3_p["<<S_id + 2<<"];\n";
      code<<"  const int3 &sp = int3_p["<<S_id + 3<<"];\n\n";

      code<<__point.str();
      switch(dt) {
        case DATA_INT:
            code<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";
          break;
        case DATA_FLOAT:
            code<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";
          break;
        case DATA_DOUBLE:
            code<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";
          break;    
      }



      /*
        start debug
      */
      // code<<"printf(\"%d\\n\", I_0);\n";
      // code<<"printf(\"S0 %d %d %d\\n\", S0[0], S0[1], S0[2]);\n";
      // code<<"printf(\"S1 %d %d %d\\n\", S1[0], S1[1], S1[2]);\n";
      // code<<"printf(\"S2 %d %d %d\\n\", S2[0], S2[1], S2[2]);\n";
      // code<<"printf(\"S3 %d %d %d\\n\", S3[0], S3[1], S3[2]);\n";
      // code<<"printf(\"S4 %d %d %d\\n\", S4[0], S4[1], S4[2]);\n";
      // code<<"printf(\"lbound %d %d %d\\n\", lbound[0], lbound[1], lbound[2]);\n";
      // code<<"printf(\"rbound %d %d %d\\n\", rbound[0], rbound[1], rbound[2]);\n";
      // code<<"printf(\"sp %d %d %d\\n\", sp[0], sp[1], sp[2]);\n";
      /*
        end debug
      */
      
      code<<"  int ist=o ; ";
      code<<"  int ied=o + sp[0] ;\n";
      code<<"  int jst=o ; ";
      code<<"  int jed=o + sp[1] ;\n";
      code<<"  int kst=o ; ";
      code<<"  int ked=o + sp[2] ;\n";

      code<<"  /*for (int kk = kst; kk< ked+BLOCK_NUM; kk += BLOCK_NUM)*/{\n";
      code<<"    //int kend=min(kk+BLOCK_NUM,ked);\n";
      code<<"    /*for (int jj = jst; jj< jed+BLOCK_NUM; jj += BLOCK_NUM)*/{\n";
      code<<"      //int jend=min(jj+BLOCK_NUM,jed);\n";
      code<<"      /*for (int ii = ist; ii< ied+BLOCK_NUM; ii += BLOCK_NUM)*/{\n";
      code<<"        //int iend=min(ii+BLOCK_NUM,ied);\n";
      code<<"        for (int k = kst; k < ked; k++) {\n";
      code<<"          for (int j = jst; j < jed; j++) {\n";
      code<<"            #pragma simd\n";
      code<<"            #pragma clang loop vectorize(assume_safety)\n";
      code<<"            #pragma clang loop interleave(enable)\n";
      code<<"            #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"            for (int i = ist; i < ied ;i++){\n";

      code<<"              list_"<<id<<"[calc_id2(i,j,k,S"<<S_id<<"_0,S"<<S_id<<"_1)] = ";

      // code<<__code.str()<<";\n   printf(\"##:%d %d %d\\n\", o, o + lbound[2], o + sp[2] - rbound[2]);  \n  }\n    }\n  }\n  return ;\n}}";

      code<<__code.str()<<";\n            }\n          }\n        }\n      }\n    }\n  }\n  return ;\n}}";      
    }
    

  }
}
