#ifndef __MIN_MAX_KERNEL_HPP__
#define __MIN_MAX_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../MPI.hpp"
#include "internal.hpp"
#include <vector>
using namespace std;


namespace oa {
  namespace kernel {

    // ap = {max/min}u
    ///:mute
    ///:include "../../NodeType.fypp"
    ///:endmute

    ///:for k in [i for i in L if i[3] in ['E']]
    ArrayPtr kernel_${k[1]}$(vector<ArrayPtr> &ops_ap);
    ///:endfor

    ///:for k in [i for i in L if i[3] == 'I']
    ///:set name = k[1]
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);
    ///:endfor
    
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
    ///:set op = 'std::abs'
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
        // oa::internal::buffer_${kernel_name}$_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
        oa::internal::buffer_${kernel_name}$_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
        
        local.pos[0] += u->get_local_box().xs();
        local.pos[1] += u->get_local_box().ys();
        local.pos[2] += u->get_local_box().zs();
        
      }else{
        local.pos[0]=local.pos[1]=local.pos[2]=-1;
      }

      MPI_Comm comm = u->get_partition()->get_comm();

      // const int size = oa::utils::get_size(comm);
      // const int rank = oa::utils::get_rank(comm);

      const int rank = MPI::global()->rank(comm);
      const int size = MPI::global()->size(comm);
      
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

          if(${op}$(global_all[i].value) ${sy}$ global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ///:if pos_mode == True
      ap = oa::funcs::ones(MPI_COMM_SELF, {{3,1,1}}, 0, DATA_INT);
      int* p = (int*)ap->get_buffer();
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
      ///:else
      ap = oa::funcs::get_seq_scalar(global.value);
      ///:endif

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }

    ///:endfor


    ///:for k in [i for i in L if i[3] == 'I']
    ///:set name = k[1]
    ///:set kernel_name = k[2]
    // A = ${sy}$ U
    template<typename TC, typename TA, typename TB>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];      

      ArrayPtr a = ArrayPool::global()->get(
          u->get_partition(),
          oa::utils::to_type<TC>());
      
      if(!a->get_partition()->equal(v->get_partition())){
        ArrayPtr v1 = oa::funcs::transfer(v, a->get_partition());
        v = v1;
      }

      const int sw_u = u->get_partition()->get_stencil_width();
      const int sw_v = v->get_partition()->get_stencil_width();
      const int sw_a = a->get_partition()->get_stencil_width();
      
      const Box bu(sw_u, u->buffer_shape()[0] - sw_u,
              sw_u, u->buffer_shape()[1] - sw_u,
              sw_u, u->buffer_shape()[2] - sw_u);

      const Box bv(sw_v, v->buffer_shape()[0] - sw_v,
              sw_v, v->buffer_shape()[1] - sw_v,
              sw_v, v->buffer_shape()[2] - sw_v);

      const Box ba(sw_a, a->buffer_shape()[0] - sw_a,
              sw_a, a->buffer_shape()[1] - sw_a,
              sw_a, a->buffer_shape()[2] - sw_a);


      if(u->has_local_data()){
        oa::internal::buffer_${kernel_name}$<TC, TA, TB>(
            (TC*)a->get_buffer(),
            (TA*)u->get_buffer(),
            (TB*)v->get_buffer(),
            a->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            ba,bu,bv);
      }

      return a;
    }
    ///:endfor
  }
}
#endif
