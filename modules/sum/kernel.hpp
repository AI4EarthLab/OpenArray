#ifndef __SUM_KERNEL_HPP__
#define __SUM_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../MPI.hpp"
#include "internal.hpp"
#include "../../utils/calcTime.hpp"
#include <vector>
using namespace std;


namespace oa {
  namespace kernel {

    ArrayPtr kernel_sum(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_csum(vector<ArrayPtr> &ops_ap);

    // salar = sum_scalar(A) 
    template <typename T>
    ArrayPtr t_kernel_sum_scalar(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      int u_dt = u->get_data_type();
      int sw = u->get_partition()->get_stencil_width();

      MPI_Comm comm = u->get_partition()->get_comm();
      const int rankID = MPI::global()->rank(comm);
      const int mpisize = MPI::global()->size(comm);

      T temp1,temp2;
      T *local_sum = &temp1;
      T *all_sum = &temp2;
      oa::internal::buffer_sum_scalar_const(
          (T*) local_sum, 
          (T*) u->get_buffer(),
          u->get_local_box(),
          sw,
          u->buffer_size());

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

      const int rankID = MPI::global()->rank(comm);
      const int mpisize = MPI::global()->size(comm);

      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
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
                  type);

              if(i+1 < sp[0])
                MPI_Send(buffer, buffersize, mpidt, receid, 0, comm);
            }
            if(rankID == receid)
              MPI_Recv(buffer, buffersize,
                      mpidt, sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        //MPI_Barrier(comm);
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
      const int rankID = MPI::global()->rank(comm);
      const int mpisize = MPI::global()->size(comm);

      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
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
                  type);

              if(j + 1 < sp[1])
                MPI_Send(buffer, buffersize, mpidt, receid, 0, comm);
            }
            if(rankID == receid)
              MPI_Recv(buffer, buffersize, mpidt,
                      sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        //MPI_Barrier(comm);
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

      const int rankID = MPI::global()->rank(comm);
      const int mpisize = MPI::global()->size(comm);

      MPI_Datatype mpidt = oa::utils::mpi_datatype(u_dt);

      PartitionPtr upar = u->get_partition();
      ap = ArrayPool::global()->get(upar, u_dt);
      ap ->set_zeros();
      Shape sp = upar->procs_shape();

      vector<int> vi = upar->get_procs_3d(rankID);

      int xs, xe, ys, ye, zs, ze;
      u->get_local_box().get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
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
              MPI_Recv(buffer, buffersize,
                      mpidt, sendid, 0, comm, MPI_STATUS_IGNORE);

          }
        //MPI_Barrier(comm);
      }

      delete [] buffer;
      return ap;
    }
  }
}
#endif
