#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
#include "Internal.hpp"
#include <vector>
using namespace std;

///:include "NodeType.fypp"
///:for m in MODULES
#include "modules/${m}$/kernel.hpp"
///:endfor

namespace oa {
  namespace kernel {

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] == 'D' or i[3] == 'E' or i[3] =='G' or i[3] == '']
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // return ANS = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);

    ///:endfor

    // ap = {max/min}u
    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] == 'E']
    ///:set name = k[1]
    ///:set kernel_name = k[2]

    ///:set sy = ">"
    ///:if k[2][-3:] == 'min'
    ///:set sy = "<"
    ///:endif

    ///:mute
    ///:set pos_mode = False
    ///:if name[-2:]=='at'
    ///:set pos_mode = True
    ///:endif

    ///:set op = ""
    ///:if name[:3] =='abs'
    ///:set op = abs
    ///:endif
    ///:endmute

    // A = ${sy}$ U
    template<typename T>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int dt = u_dt;
      int sw = u->get_partition()->get_stencil_width();

      typedef struct {
        T value;
        int pos[3];
      } m_info;

      m_info local, global;
      if(u->has_local_data()){
        oa::internal::buffer_${kernel_name}$_const(local.value,
                                                   local.pos,
                                                   (T*) u->get_buffer(),
                                                   u->get_local_box(),
                                                   sw);
      }else{
        local.pos[0]=local.pos[1]=local.pos[2]=-1;
      }

      MPI_Comm comm = u->get_partition()->get_comm();

      const int size = oa::utils::get_size(comm);
      const int rank = oa::utils::get_rank(comm);

      m_info* global_all = new m_info[size];

      MPI_Gather(&local,
                 sizeof(m_info),
                 MPI_BYTE,
                 global_all,
                 sizeof(m_info),
                 MPI_BYTE,
                 0,
                 comm);

      // std::cout<<"rank="<<rank
      //          <<" "<<local.value <<" "
      //          <<local.pos[0]<<" "
      //          <<local.pos[1]<<" "
      //          <<local.pos[2]<<" "
      //          <<"sizeof="<<sizeof(m_info)<<std::endl;

      //u->get_local_box().display("b");

      if(rank == 0){
        global = local;
        for(int i = 0; i < size; ++i){

          // std::cout<<"rank="<<rank
          //          <<" global:"<<global_all[i].value <<" "
          //          <<global_all[i].pos[0]<<" "
          //          <<global_all[i].pos[1]<<" "
          //          <<global_all[i].pos[2]<<std::endl;
          if(global_all[i].pos[0] < 0) continue;

          if(global_all[i].value ${sy}$ global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ///:if pos_mode == True
      ap = oa::funcs::ones(MPI_COMM_SELF, {3,1,1}, 0, DATA_INT);
      int* p = (int*)ap->get_buffer();
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
      ///:else
      ap = oa::funcs::get_seq_scalar(global.value);
      // if(rank == 0)
      //   ap->display("ap = ");
      ///:endif

      return ap;
    }

    ///:endfor

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] == 'D']
    ///:set name = k[1]
    // crate kernel_${name}$
    // A = ${name}$(U)
    template<typename T>
    void t_kernel_${name}$(ArrayPtr& ap, ArrayPtr& u) {
      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      int3 lbound = {0, 0, 0};
      int3 rbound = {0, 0, 0};
      ///:if name[1:] == "xb"
      lbound = {1, 0, 0};
      ///:elif name[1:] == "xf"
      rbound = {1, 0, 0};
      ///:elif name[1:] == "yb"
      lbound = {0, 1, 0};
      ///:elif name[1:] == "yf"
      rbound = {0, 1, 0};
      ///:elif name[1:] == "zb"
      lbound = {0, 0, 1};
      ///:elif name[1:] == "zf"
      rbound = {0, 0, 1};
      ///:endif

      double* ans = (double*) ap->get_buffer();
      T* buffer = (T*) u->get_buffer();

      vector<MPI_Request> reqs;
      oa::funcs::update_ghost_start(u, reqs, -1);
      oa::internal::${name}$_calc_inside<T>(ans, buffer, lbound, rbound, sw, sp, S);
      oa::funcs::update_ghost_end(reqs);
      oa::internal::${name}$_calc_outside<T>(ans, buffer, lbound, rbound, sw, sp, S);

    }

    ///:endfor

  }
}
#endif
