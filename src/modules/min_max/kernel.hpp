#ifndef __MIN_MAX_KERNEL_HPP__
#define __MIN_MAX_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../MPI.hpp"
#include "internal.hpp"
#ifdef __HAVE_CUDA__
  #include "internalGpu.hpp"
#endif
#include <vector>
using namespace std;


namespace oa {
  namespace kernel {

    // ap = {max/min}u

    ArrayPtr kernel_min(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_max(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_min_at(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_max_at(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_abs_max(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_abs_min(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_abs_max_at(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_abs_min_at(vector<ArrayPtr> &ops_ap);

    ArrayPtr kernel_min2(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_max2(vector<ArrayPtr> &ops_ap);
    



    // A = < U
    template<typename T>
    ArrayPtr t_kernel_min(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_min_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if((global_all[i].value) < global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      //GPU WARNING
      ap = oa::funcs::get_seq_scalar(global.value);

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = > U
    template<typename T>
    ArrayPtr t_kernel_max(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_max_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if((global_all[i].value) > global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      //GPU WARNING
      ap = oa::funcs::get_seq_scalar(global.value);

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = < U
    template<typename T>
    ArrayPtr t_kernel_min_at(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_min_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if((global_all[i].value) < global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ap = oa::funcs::ones(MPI_COMM_SELF, {{3,1,1}}, 0, DATA_INT);
#ifndef __HAVE_CUDA__
      int* p = (int*)ap->get_buffer();
#else
      int* p = (int*)ap->get_cpu_buffer();
#endif
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = > U
    template<typename T>
    ArrayPtr t_kernel_max_at(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_max_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if((global_all[i].value) > global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ap = oa::funcs::ones(MPI_COMM_SELF, {{3,1,1}}, 0, DATA_INT);
#ifndef __HAVE_CUDA__
      int* p = (int*)ap->get_buffer();
#else
      int* p = (int*)ap->get_cpu_buffer();
#endif
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = > U
    template<typename T>
    ArrayPtr t_kernel_abs_max(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_abs_max_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_abs_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_abs_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if(std::abs(global_all[i].value) > global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      //GPU WARNING
      ap = oa::funcs::get_seq_scalar(global.value);

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = < U
    template<typename T>
    ArrayPtr t_kernel_abs_min(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_abs_min_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_abs_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_abs_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if(std::abs(global_all[i].value) < global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      //GPU WARNING
      ap = oa::funcs::get_seq_scalar(global.value);

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = > U
    template<typename T>
    ArrayPtr t_kernel_abs_max_at(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_abs_max_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_abs_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_abs_max_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if(std::abs(global_all[i].value) > global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ap = oa::funcs::ones(MPI_COMM_SELF, {{3,1,1}}, 0, DATA_INT);
#ifndef __HAVE_CUDA__
      int* p = (int*)ap->get_buffer();
#else
      int* p = (int*)ap->get_cpu_buffer();
#endif
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }




    // A = < U
    template<typename T>
    ArrayPtr t_kernel_abs_min_at(vector<ArrayPtr> &ops_ap) {
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
        // oa::internal::buffer_abs_min_const(local.value,
        //         local.pos,
        //         (T*) u->get_buffer(),
        //         u->get_local_box(),
        //         sw);
#ifndef __HAVE_CUDA__
        oa::internal::buffer_abs_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#else
        oa::internal::gpu::buffer_abs_min_const(local.value,
                local.pos,
                (T*) u->get_buffer(),
                u->buffer_shape(),
                u->local_data_win());
#endif
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

          if(std::abs(global_all[i].value) < global.value){
            global = global_all[i];
          }
        }
      }

      MPI_Bcast(&global, sizeof(m_info), MPI_BYTE, 0, comm);

      delete(global_all);

      ArrayPtr ap;

      ap = oa::funcs::ones(MPI_COMM_SELF, {{3,1,1}}, 0, DATA_INT);
#ifndef __HAVE_CUDA__
      int* p = (int*)ap->get_buffer();
#else
      int* p = (int*)ap->get_cpu_buffer();
#endif
      p[0] = global.pos[0];
      p[1] = global.pos[1];
      p[2] = global.pos[2];
#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif

      // if(rank == 0)
      //   ap->display("ap = ");

      return ap;
    }



    // A = < U
    template<typename TC, typename TA, typename TB>
    ArrayPtr t_kernel_min2(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];      

      ArrayPtr a = ArrayPool::global()->get(
          u->get_partition(),
          oa::utils::to_type<TC>());
      
      if(!a->get_partition()->equal(v->get_partition())){
        //GPU WARNING
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
#ifndef __HAVE_CUDA__
        oa::internal::buffer_min2<TC, TA, TB>(
            (TC*)a->get_buffer(),
            (TA*)u->get_buffer(),
            (TB*)v->get_buffer(),
            a->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            ba,bu,bv);
#else
        oa::internal::gpu::buffer_min2<TC, TA, TB>(
            (TC*)a->get_buffer(),
            (TA*)u->get_buffer(),
            (TB*)v->get_buffer(),
            a->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            ba,bu,bv);
#endif
      }

      return a;
    }
    // A = < U
    template<typename TC, typename TA, typename TB>
    ArrayPtr t_kernel_max2(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];      

      ArrayPtr a = ArrayPool::global()->get(
          u->get_partition(),
          oa::utils::to_type<TC>());
      
      if(!a->get_partition()->equal(v->get_partition())){
        //GPU WARNING
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
#ifndef __HAVE_CUDA__
        oa::internal::buffer_max2<TC, TA, TB>(
            (TC*)a->get_buffer(),
            (TA*)u->get_buffer(),
            (TB*)v->get_buffer(),
            a->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            ba,bu,bv);
#else
        oa::internal::gpu::buffer_max2<TC, TA, TB>(
            (TC*)a->get_buffer(),
            (TA*)u->get_buffer(),
            (TB*)v->get_buffer(),
            a->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            ba,bu,bv);
#endif
      }

      return a;
    }
  }
}
#endif
