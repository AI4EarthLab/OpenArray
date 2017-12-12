#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
#include "Internal.hpp"
#include <vector>
using namespace std;

namespace oa {
  namespace kernel {

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L]
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // return ANS = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);

    ///:endfor
    
    // ap = u {+ - * /} v
    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] == 'A']
    ///:set name = k[1]
    ///:set sy = k[2]
    // A = U ${sy}$ V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      // support pseudo array calculation
      if (u->is_pseudo() || v->is_pseudo()) {
        int su = oa::utils::get_shape_dimension(u->local_shape());
        int sv = oa::utils::get_shape_dimension(v->local_shape());

        PartitionPtr pp;
        if (su > sv) pp = u->get_partition();
        else pp = v->get_partition(); 
        
        ap = ArrayPool::global()->get(pp, dt);

        oa::internal::pseudo_buffer_${name}$_buffer(
          (T1*) ap->get_buffer(),
          (T2*) u->get_buffer(),
          (T3*) v->get_buffer(),
          ap->get_local_box(),
          u->get_local_box(),
          v->get_local_box(),
          ap->buffer_shape(),
          u->buffer_shape(),
          v->buffer_shape(),
          pp->get_stencil_width()
        );

      } else {
        if (u->is_seqs_scalar()) {
          // (1) u is a scalar
          ap = ArrayPool::global()->get(v->get_partition(), dt);
          T2 scalar = *(T2*) u->get_buffer();
          oa::internal::const_${name}$_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_size()
          );
        } else if (v->is_seqs_scalar()) {
          // (2) v is a scalar
          ap = ArrayPool::global()->get(u->get_partition(), dt);
          T3 scalar = *(T3*) v->get_buffer();
          oa::internal::buffer_${name}$_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            ap->buffer_size()
          );
        } else {
          PartitionPtr upar = u->get_partition();
          PartitionPtr vpar = v->get_partition();
          assert(upar->get_comm() == vpar->get_comm());

  /*        // U and V must have same shape
          assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
  */
          ap = ArrayPool::global()->get(upar, dt);
          if (upar->equal(vpar)) {
            oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_size()
            );
          } else {
            ArrayPtr tmp = oa::funcs::transfer(v, upar);
            oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_size()
            );
          }
        }
      }

      return ap;
    }

    ///:endfor
  
    ///:mute
    ///:set K = [['gt','>'], ['ge', '>='], ['lt', '<'],['le', '<='], &
                 ['eq','=='], ['ne','/='],['and','&&'],['or','||']]
    ///:endmute
    ///:for t in K
    ///:set name = t[0]
    ///:set sy = t[1]
    // A = U ${sy}$ V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_${name}$_buffer(
          (T1 *) ap->get_buffer(),
          scalar,
          (T3 *) v->get_buffer(),
          //ap->buffer_size()
          ap->buffer_shape(),
          ap->get_partition()->get_stencil_width()
        );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_${name}$_const(
          (T1 *) ap->get_buffer(),
          (T2 *) u->get_buffer(),
          scalar,
          //ap->buffer_size()
          ap->buffer_shape(),
          ap->get_partition()->get_stencil_width()
        );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

/*        // U and V must have same shape
        assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
*/
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_${name}$_buffer(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
          );
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_${name}$_buffer(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            (T3 *) tmp->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
          );
        }
      }
      return ap;
    }

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

    // salar = sum_scalar(A) 
    template <typename T>
    ArrayPtr t_kernel_sum_scalar(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int sw = u->get_partition()->get_stencil_width();

      MPI_Comm comm = u->get_partition()->get_comm();
      int rankID = oa::utils::get_rank(comm);
      int mpisize =  oa::utils::get_size(comm);

      T temp1,temp2;
      T *local_sum = &temp1;
      T *all_sum = &temp2;
      oa::internal::buffer_sum_scalar_const(
          (T*) local_sum, 
          (T*) u->get_buffer(),
          u->get_local_box(),
          sw,
          u->buffer_size()
          );

      //std::cout<<"mpi"<<rankID<<" local_sum ="<<*local_sum<<std::endl;
      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);
      MPI_Allreduce(local_sum, all_sum, 1, mpidt, MPI_SUM, comm);
      ArrayPtr ap = oa::funcs::get_seq_scalar(*all_sum);
      //std::cout << "The sum is: " << *all_sum << std::endl;
      return ap;
    }

    //sum to x 
    template <typename T>
    ArrayPtr t_kernel_csum_x(vector<ArrayPtr> &ops_ap) {
      ArrayPtr ap;
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int sw = u->get_partition()->get_stencil_width();

      MPI_Comm comm = u->get_partition()->get_comm();
      int rankID = oa::utils::get_rank(comm);
      int mpisize = oa::utils::get_size(comm);
      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners(xs, xe, ys, ye, zs, ze, sw);
      int buffersize = (ye-ys-2*sw)*(ze-zs-2*sw);
      T * buffer = new T[buffersize];

      for(int i = 0; i < sp[0]; i++)
      {
        int type;  //type:   top 2  mid 1  bottom 0
        if(i == 0) 
          type = 2;
        else if(i == sp[0] - 1) 
          type = 0;
        else
          type = 1;

        for(int j = 0; j < sp[1]; j++)
          for(int k = 0; k < sp[2]; k++){
            int sendid = upar->get_procs_rank(i, j, k);
            int receid = -1;
            if(i+1 < sp[0])
              receid = upar->get_procs_rank(i+1, j, k);
            if(rankID == sendid){
              oa::internal::buffer_csum_x_const(
                  (T*) ap->get_buffer(),
                  (T*) u->get_buffer(),
                  u->get_local_box(),
                  sw,
                  u->buffer_size(),
                  buffer,
                  type
                  );

              if(i+1 < sp[0])
                MPI_Send(buffer, buffersize, mpidt, receid, 0, comm);
            }
            if(rankID == receid)
              MPI_Recv(buffer, buffersize, mpidt, sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        MPI_Barrier(comm);
      }

      delete []buffer;
      return ap;
    }

    //csum to y
    template <typename T>
    ArrayPtr t_kernel_csum_y(vector<ArrayPtr> &ops_ap) {
      ArrayPtr ap;
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int sw = u->get_partition()->get_stencil_width();

      MPI_Comm comm = u->get_partition()->get_comm();
      int rankID = oa::utils::get_rank(comm);
      int mpisize = oa::utils::get_size(comm);
      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners(xs, xe, ys, ye, zs, ze, sw);
      int buffersize = (xe-xs-2*sw)*(ze-zs-2*sw);
      T * buffer = new T[buffersize];

      for(int j = 0; j < sp[1]; j++)
      {
        int type;  //type:   top 2  mid 1  bottom 0
        if(j == 0) 
          type = 2;
        else if(j == sp[1]-1) 
          type = 0;
        else
          type = 1;

        for(int i = 0; i < sp[0]; i++)
          for(int k = 0; k < sp[2]; k++){
            int sendid = upar->get_procs_rank(i, j, k);
            int receid = -1;
            if(j + 1 < sp[1])
              receid = upar->get_procs_rank(i, j+1, k);
            if(rankID == sendid){
              oa::internal::buffer_csum_y_const(
                  (T*) ap->get_buffer(),
                  (T*) u->get_buffer(),
                  u->get_local_box(),
                  sw,
                  u->buffer_size(),
                  buffer,
                  type
                  );

              if(j + 1 < sp[1])
                MPI_Send(buffer, buffersize, mpidt, receid, 0, comm);
            }
            if(rankID == receid)
              MPI_Recv(buffer, buffersize, mpidt, sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        MPI_Barrier(comm);
      }

      delete []buffer;
      return ap;
    }


    //csum to z
    template <typename T>
    ArrayPtr t_kernel_csum_z(vector<ArrayPtr> &ops_ap) {
      ArrayPtr ap;
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int sw = u->get_partition()->get_stencil_width();

      MPI_Comm comm = u->get_partition()->get_comm();
      int rankID = oa::utils::get_rank(comm);
      int mpisize = oa::utils::get_size(comm);
      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners(xs, xe, ys, ye, zs, ze, sw);
      int buffersize = (xe-xs-2*sw)*(ye-ys-2*sw);
      T * buffer = new T[buffersize];

      for(int k = 0; k < sp[2]; k++)
      {
        int type;  //type:   top 2  mid 1  bottom 0
        if(k == 0) 
          type = 2;
        else if(k == sp[2]-1) 
          type = 0;
        else
          type = 1;

        for(int i = 0; i < sp[0]; i++)
          for(int j = 0; j < sp[1]; j++){
            int sendid = upar->get_procs_rank(i, j, k);
            int receid = -1;
            if(k + 1 < sp[2])
              receid = upar->get_procs_rank(i, j, k + 1);
            if(rankID == sendid){
              oa::internal::buffer_csum_z_const(
                  (T*) ap->get_buffer(),
                  (T*) u->get_buffer(),
                  u->get_local_box(),
                  sw,
                  u->buffer_size(),
                  buffer,
                  type
                  );
              if(k + 1 < sp[2])
                MPI_Send(buffer, buffersize, mpidt, receid, 0, comm);
            }
            if(rankID == receid)
              MPI_Recv(buffer, buffersize, mpidt, sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        MPI_Barrier(comm);
      }

      delete []buffer;
      return ap;
    }

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
