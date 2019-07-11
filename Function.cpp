/*
 * Function.cpp
 *
=======================================================*/

#include "MPI.hpp"
#include "Kernel.hpp"
#include "common.hpp"
#include "Function.hpp"
#include "utils/utils.hpp"
#include "utils/calcTime.hpp"
#include "GpuKernels.hpp"
#include <mpi.h>
#include <fstream>

using namespace std;
namespace oa {
  namespace funcs {


    // create a ones array
    ArrayPtr ones(MPI_Comm comm, const Shape& s, 
                  int stencil_width, int data_type) {
      ArrayPtr ap;
      switch(data_type) {
      case DATA_INT:
        ap = consts(comm, s, (int)1, stencil_width);
        break;
      case DATA_FLOAT:
        ap = consts(comm, s, (float)1, stencil_width);
        break;
      case DATA_DOUBLE:
        ap = consts(comm, s, (double)1, stencil_width);
        break;
      }   
      return ap;
    }

    // create a zeros array
    ArrayPtr zeros(MPI_Comm comm, const Shape& s, 
                   int stencil_width, int data_type) {
      ArrayPtr ap;
      switch(data_type) {
      case DATA_INT:
        ap = consts(comm, s, (int)0, stencil_width);
        break;
      case DATA_FLOAT:
        ap = consts(comm, s, (float)0, stencil_width);
        break;
      case DATA_DOUBLE:
        ap = consts(comm, s, (double)0, stencil_width);
        break;
      }
      return ap;
    }

    // create a rand array
    ArrayPtr rands(MPI_Comm comm, const Shape& s, 
                  int stencil_width, int data_type) {
      ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size_with_stencil(stencil_width);
      switch(data_type) {
      case DATA_INT:
        oa::internal::set_buffer_rand((int*)ap->get_buffer(), size);
        break;
      case DATA_FLOAT:
        oa::internal::set_buffer_rand((float*)ap->get_buffer(), size);
        break;
      case DATA_DOUBLE:
        oa::internal::set_buffer_rand((double*)ap->get_buffer(), size);
        break;
      }
      return ap;
    }

    // create a seqs array
    ArrayPtr seqs(MPI_Comm comm, const Shape& s, 
                  int stencil_width, int data_type, DeviceType dev_type) {
      #ifndef __HAVE_CUDA__
      dev_type = CPU;
      #endif;

      ArrayPtr ap = ArrayPool::global()->
        get(comm, s, stencil_width, data_type, dev_type);
      Box box = ap->get_local_box();

      #ifndef __HAVE_CUDA__
      switch(data_type) {
      case DATA_INT:
        oa::internal::set_buffer_seqs((int*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_FLOAT:
        oa::internal::set_buffer_seqs((float*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_DOUBLE:
        oa::internal::set_buffer_seqs((double*)ap->get_buffer(), s, box, stencil_width);
        break;
      }

      #else
      switch(data_type) {
      case DATA_INT:
        oa::gpu::set_buffer_seqs((int*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_FLOAT:
        oa::gpu::set_buffer_seqs((float*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_DOUBLE:
        oa::gpu::set_buffer_seqs((double*)ap->get_buffer(), s, box, stencil_width);
        break;
      }
      #endif
      return ap;
    }

    ArrayPtr seqs(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
                  const vector<int> &z, int stencil_width, int data_type, DeviceType dev_type) {
      #ifndef __HAVE_CUDA__
      dev_type = CPU;
      #endif

      ArrayPtr ap = ArrayPool::global()->
        get(comm, x, y, z, stencil_width, data_type, dev_type);
      Box box = ap->get_local_box();
      Shape s = ap->shape();

      #ifndef __HAVE_CUDA__
      switch(data_type) {
      case DATA_INT:
        oa::internal::set_buffer_seqs((int*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_FLOAT:
        oa::internal::set_buffer_seqs((float*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_DOUBLE:
        oa::internal::set_buffer_seqs((double*)ap->get_buffer(), s, box, stencil_width);
        break;
      }
      #else
      switch(data_type) {
      case DATA_INT:
        oa::gpu::set_buffer_seqs((int*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_FLOAT:
        oa::gpu::set_buffer_seqs((float*)ap->get_buffer(), s, box, stencil_width);
        break;
      case DATA_DOUBLE:
        oa::gpu::set_buffer_seqs((double*)ap->get_buffer(), s, box, stencil_width);
        break;
      }
      #endif
      return ap;
    }

    double local_sub(const ArrayPtr &ap, int x, int y, int z) {
      Box b(x, x+1, y, y+1, z, z+1);
      Box local_box = ap->get_local_box();
      //b.display("b = ");
      //local_box.display("local_box = ");
      //assert(b.is_inside(local_box));
      int sw = ap->get_stencil_width();
      x -= local_box.xs();
      y -= local_box.ys();
      z -= local_box.zs();
      double ans;
      switch(ap->get_data_type()) {
      case DATA_INT:
        ans = oa::internal::get_buffer_local_sub((int*)ap->get_buffer(), local_box, x, y, z, sw);
        break;
      case DATA_FLOAT:
        ans = oa::internal::get_buffer_local_sub((float*)ap->get_buffer(), local_box, x, y, z, sw);
        break;
      case DATA_DOUBLE:
        ans = oa::internal::get_buffer_local_sub((double*)ap->get_buffer(), local_box, x, y, z, sw);
        break;
      }
      return ans;
      
    }
    //transfer(ArrayPtr &A, ArrayPtr &B);
    ArrayPtr subarray(const ArrayPtr &ap, const Box &b) {
      vector<int> rsx, rsy, rsz;
      PartitionPtr pp = ap->get_partition();
      Shape ps = pp->procs_shape();
      pp->split_box_procs(b, rsx, rsy, rsz);
      
      vector<int> x(ps[0], 0), y(ps[1], 0), z(ps[2], 0);
      for (int i = 0; i < rsx.size(); i += 3)
        x[rsx[i + 2]] = rsx[i + 1] - rsx[i];
      for (int i = 0; i < rsy.size(); i += 3)
        y[rsy[i + 2]] = rsy[i + 1] - rsy[i];
      for (int i = 0; i < rsz.size(); i += 3)
        z[rsz[i + 2]] = rsz[i + 1] - rsz[i];
     
      #ifndef __HAVE_CUDA__
        DeviceType dev_type = CPU;
      #else 
        DeviceType dev_type = CPU_AND_GPU;
      #endif
 
      ArrayPtr arr_ptr = ArrayPool::global()->
        get(pp->get_comm(), x, y, z, pp->get_stencil_width(), ap->get_data_type(), dev_type);
      // printf("shape:%d %d %d\n", ps[0], ps[1], ps[2]);
      // for(int i = 0; i < rsx.size(); i++){
      //   printf("%d ", rsx[i]);
      // }
      // printf("\n");
      // for(int i = 0; i < rsy.size(); i++){
      //   printf("%d ", rsy[i]);
      // }
      // printf("\n");
      // for(int i = 0; i < rsz.size(); i++){
      //   printf("%d ", rsz[i]);
      // }
      // printf("\n");
      
      // printf("D:%d %d %d\n", arr_ptr->shape()[0],
      //         arr_ptr->shape()[1],
      //         arr_ptr->shape()[2]);

      // don't have local data in process
      if (!arr_ptr->has_local_data()) return arr_ptr; 
      
      int rk = pp->rank();
      vector<int> procs_coord = pp->get_procs_3d(rk);

      int idx = procs_coord[0] - rsx[2];
      int idy = procs_coord[1] - rsy[2];
      int idz = procs_coord[2] - rsz[2];

      Box box = ap->get_local_box();
      Box sub_box(
                  rsx[idx * 3], rsx[idx * 3 + 1],
                  rsy[idy * 3], rsy[idy * 3 + 1], 
                  rsz[idz * 3], rsz[idz * 3 + 1]
                  );

      // different data_type
      #ifndef __HAVE_CUDA__
      switch(ap->get_data_type()) {
      case DATA_INT:
        oa::internal::get_buffer_subarray<int>
          ((int*) arr_ptr->get_buffer(),
                (int*) ap->get_buffer(),
                sub_box, box,
                pp->get_stencil_width());
        break;
      case DATA_FLOAT:
        oa::internal::get_buffer_subarray<float>
          ((float*) arr_ptr->get_buffer(),
                  (float*) ap->get_buffer(),
                  sub_box, box, pp->get_stencil_width());
        break;
      case DATA_DOUBLE:
        oa::internal::get_buffer_subarray<double>
          ((double*) arr_ptr->get_buffer(),
                  (double*) ap->get_buffer(), 
                  sub_box, box, pp->get_stencil_width());
        break;
      }
      #else
      switch(ap->get_data_type()) {
      case DATA_INT:
        oa::gpu::get_buffer_subarray<int>
          ((int*) arr_ptr->get_buffer(),
                (int*) ap->get_buffer(),
                sub_box, box,
                pp->get_stencil_width());
        break;
      case DATA_FLOAT:
        oa::gpu::get_buffer_subarray<float>
          ((float*) arr_ptr->get_buffer(),
                  (float*) ap->get_buffer(),
                  sub_box, box, pp->get_stencil_width());
        break;
      case DATA_DOUBLE:
        oa::gpu::get_buffer_subarray<double>
          ((double*) arr_ptr->get_buffer(),
                  (double*) ap->get_buffer(), 
                  sub_box, box, pp->get_stencil_width());
        break;
      }
      #endif
      return arr_ptr;
    }
    
    //transfer src to dest based on dest's partition pp
    ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp, DeviceType dev_type) {

      #ifndef __HAVE_CUDA__
      dev_type = CPU;
      #endif
      ArrayPtr ap = ArrayPool::global()->get(pp, src->get_data_type(), dev_type);

      int sw = pp->get_stencil_width();

      int num_procs = pp->procs_size();
      MPI_Request isreqs[num_procs], irreqs[num_procs];
      int isreqs_cnt = 0, irreqs_cnt = 0;

      // src has local data, transfer to ap
      if (src->has_local_data()) {
        vector<int> rsx, rsy, rsz;
        Box src_box = src->get_local_box();
        pp->split_box_procs(src_box, rsx, rsy, rsz);

        int xs, ys, zs, xe, ye, ze;
        src_box.get_corners(xs, xe, ys, ye, zs, ze);

        /*  debug
            src_box.display();
            for (int i = 0; i < rsx.size(); i += 3) 
            printf("rsx: [%d %d %d]\n", rsx[i], rsx[i + 1], rsx[i + 2]);
            for (int i = 0; i < rsy.size(); i += 3) 
            printf("rsy: [%d %d %d]\n", rsy[i], rsy[i + 1], rsy[i + 2]);
            for (int i = 0; i < rsz.size(); i += 3) 
            printf("rsz: [%d %d %d]\n", rsz[i], rsz[i + 1], rsz[i + 2]);
        */

        vector<int> acc_rsx, acc_rsy, acc_rsz;
        pp->get_acc_box_procs(rsx, rsy, rsz,
                              acc_rsx, acc_rsy, acc_rsz);

        for (int i = 0; i < rsx.size(); i += 3) {
          if (rsx[i] == rsx[i + 1]) continue; 
          for (int j = 0; j < rsy.size(); j += 3) {
            if (rsy[j] == rsy[j + 1]) continue;
            for (int k = 0; k < rsz.size(); k += 3) {
              if (rsz[k] == rsz[k + 1]) continue;

              MPI_Datatype src_subarray;
              int starts[3]  = {
                sw + acc_rsx[i / 3], 
                sw + acc_rsy[j / 3], 
                sw + acc_rsz[k / 3]};
              int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
              int subsize[3] = {
                rsx[i + 1] - rsx[i], 
                rsy[j + 1] - rsy[j], 
                rsz[k + 1] - rsz[k]
              };
              
              /*  debug
                  vector<int> cd = pp->get_procs_3d(pp->rank());
                  printf("====Send====\n");
                  printf("from process [%d %d %d]\n", cd[0], cd[1], cd[2]);
                  printf("to process   [%d %d %d]\n", rsx[i + 2], rsy[j + 2], rsz[k + 2]);
                  printf("starts  [%d %d %d]\n", starts[0], starts[1], starts[2]);
                  printf("bigsize [%d %d %d]\n", bigsize[0], bigsize[1], bigsize[2]);
                  printf("subsize [%d %d %d]\n\n", subsize[0], subsize[1], subsize[2]);
              */

              MPI_Type_create_subarray(3, bigsize, subsize,
                                       starts, MPI_ORDER_FORTRAN,
                                       oa::utils::mpi_datatype(src->get_data_type()),
                                       &src_subarray);
              MPI_Type_commit(&src_subarray);

              int target_rank = pp->
                get_procs_rank({rsx[i + 2], rsy[j + 2], rsz[k + 2]});
              MPI_Isend(src->get_buffer(), 1, 
                        src_subarray, target_rank, 
                        100, pp->get_comm(),
                        &isreqs[isreqs_cnt++]);
              MPI_Type_free(&src_subarray);

            }
          }
        }
      }

      // ap has local data, receive from other processes
      if (ap->has_local_data()) {
        vector<int> rsx, rsy, rsz;
        Box dest_box = ap->get_local_box();
        src->get_partition()->split_box_procs(dest_box, rsx, rsy, rsz);

        int xs, ys, zs, xe, ye, ze;
        dest_box.get_corners(xs, xe, ys, ye, zs, ze);

        vector<int> acc_rsx, acc_rsy, acc_rsz;
        pp->get_acc_box_procs(rsx, rsy, rsz,
                              acc_rsx, acc_rsy, acc_rsz);

        for (int i = 0; i < rsx.size(); i += 3) {
          if (rsx[i] == rsx[i + 1]) continue;
          for (int j = 0; j < rsy.size(); j += 3) {
            if (rsy[j] == rsy[j + 1]) continue;
            for (int k = 0; k < rsz.size(); k += 3) {
              if (rsz[k] == rsz[k + 1]) continue;
              MPI_Datatype dest_subarray;
              int starts[3]  = {
                sw + acc_rsx[i / 3], 
                sw + acc_rsy[j / 3], 
                sw + acc_rsz[k / 3]};
              int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
              int subsize[3] = {
                rsx[i + 1] - rsx[i], 
                rsy[j + 1] - rsy[j], 
                rsz[k + 1] - rsz[k]
              };

              /* debug
                 vector<int> cd = pp->get_procs_3d(pp->rank());
                 printf("====Receive====\n");
                 printf("to process   [%d %d %d]\n", cd[0], cd[1], cd[2]);
                 printf("from process [%d %d %d]\n", rsx[i + 2], rsy[j + 2], rsz[k + 2]);
                 printf("starts  [%d %d %d]\n", starts[0], starts[1], starts[2]);
                 printf("bigsize [%d %d %d]\n", bigsize[0], bigsize[1], bigsize[2]);
                 printf("subsize [%d %d %d]\n\n", subsize[0], subsize[1], subsize[2]);
              */
              
              MPI_Type_create_subarray(3, bigsize, subsize,
                                       starts, MPI_ORDER_FORTRAN,
                                       oa::utils::mpi_datatype(ap->get_data_type()),
                                       &dest_subarray);
              MPI_Type_commit(&dest_subarray);

              int target_rank = src->get_partition()->
                get_procs_rank({rsx[i + 2], rsy[j + 2], rsz[k + 2]});
              MPI_Irecv(ap->get_buffer(), 1,
                        dest_subarray, target_rank, 
                        100, pp->get_comm(),
                        &irreqs[irreqs_cnt++]);

              MPI_Type_free(&dest_subarray);
              
            }
          }
        } 
      }

      MPI_Waitall(isreqs_cnt, &isreqs[0], MPI_STATUSES_IGNORE);
      MPI_Waitall(irreqs_cnt, &irreqs[0], MPI_STATUSES_IGNORE);
      return ap;
    }

    void update_ghost(ArrayPtr ap, DeviceType dev_type) {
      vector<MPI_Request> reqs;
      update_ghost_start(ap, reqs, -1, {{0,0,0}}, {{0,0,0}}, dev_type);
      update_ghost_end(reqs);
    }
    
    oa_int3 get_update_ghost(oa_int3 bound, bool x, bool y, bool z) {
      if (x == true) bound[0] = 0;
      if (y == true) bound[1] = 0;
      if (z == true) bound[2] = 0;
      return bound;
    }

    void update_ghost_start(ArrayPtr ap, vector<MPI_Request> &reqs, int direction, oa_int3 lbound, oa_int3 rbound, DeviceType dev_type) {
      // set ghost to zeros, then eval's answer equal to eval_with_op      
      // set_ghost_zeros(ap); 


int cta=0,ctb=0;
if(lbound[0]) cta++;
if(lbound[1]) cta++;
if(lbound[2]) cta++;
if(rbound[0]) cta++;
if(rbound[1]) cta++;
if(rbound[2]) cta++;


      oa_int3 lb = lbound;
      oa_int3 rb = rbound;
      //oa_int3 lb = get_update_ghost(lbound, ap->get_lb_ghost_updated(0), ap->get_lb_ghost_updated(1), ap->get_lb_ghost_updated(2));
      //oa_int3 rb = get_update_ghost(rbound, ap->get_rb_ghost_updated(0), ap->get_rb_ghost_updated(1), ap->get_rb_ghost_updated(2));

if(lb[0]) ctb++;
if(lb[1]) ctb++;
if(lb[2]) ctb++;
if(rb[0]) ctb++;
if(rb[1]) ctb++;
if(rb[2]) ctb++;
//if(cta != ctb) printf("===: %d ,%d \n",cta,ctb);
      ap->update_lb_ghost_updated(lbound);
      ap->update_rb_ghost_updated(rbound);
      
      PartitionPtr pp = ap->get_partition();
      Shape arr_shape = ap->shape();
      int gx = arr_shape[0];
      int gy = arr_shape[1];
      int gz = arr_shape[2];

      Shape procs_shape = pp->procs_shape();
      int px = procs_shape[0];
      int py = procs_shape[1];
      int pz = procs_shape[2];

      Shape bt_shape = pp->get_bound_type();
      int bx = bt_shape[0];
      int by = bt_shape[1];
      int bz = bt_shape[2];

      int s = pp->get_stencil_width();
      int st = pp->get_stencil_type();

      vector<int> coord = pp->get_procs_3d(pp->rank());
      int xs, ys, zs, xe, ye, ze;
      pp->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
      //printf("**:%d %d %d %d %d %d\n", xs, xe, ys, ye, zs, ze);
      
      int o_cx[3] = {0, s, s + xe - xs};
      int o_cy[3] = {0, s, s + ye - ys};
      int o_cz[3] = {0, s, s + ze - zs};

      int o_dx[3] = {s, xe - xs, s};
      int o_dy[3] = {s, ye - ys, s};
      int o_dz[3] = {s, ze - zs, s}; 

      int i_cx[3] = {0, 0, xe - xs - s};
      int i_cy[3] = {0, 0, ye - ys - s};
      int i_cz[3] = {0, 0, ze - zs - s};

      int i_dx[3] = {s, xe - xs, s};
      int i_dy[3] = {s, ye - ys, s};
      int i_dz[3] = {s, ze - zs, s};

      int total_send_size = 0;
      int neighbour_procs[27][3];
      bool update_bound_send[27] = {false};
      bool update_bound_recv[27] = {false};

      Box i_box[27], o_box[27];

      int update_cnt = 0;

      if (!ap->has_local_data()) return ;

      int st_x, st_y, st_z, ed_x, ed_y, ed_z;
      st_x = st_y = st_z = ed_x = ed_y = ed_z = 0;

      bitset<3> bit = ap->get_bitset();

      switch (direction) {
      case 0:
        st_x = -1, ed_x = 1;
        break;
      case 1:
        st_y = -1, ed_y = 1;
        break;
      case 2:
        st_z = -1, ed_z = 1;
        break;
      case -1:
        st_x = st_y = st_z = -1;
        ed_x = ed_y = ed_z = 1;
        break;
      case 3:
        if (bit[2] != 0) st_x = -1, ed_x = 1;
        if (bit[1] != 0) st_y = -1, ed_y = 1;
        if (bit[0] != 0) st_z = -1, ed_z = 1;
        break;
      case 4:
        st_x = -rb[0], ed_x = lb[0];
        st_y = -rb[1], ed_y = lb[1];
        st_z = -rb[2], ed_z = lb[2];
        break;
      }

      // update_bound_send
      for (int z = st_z; z <= ed_z; ++z) {
        for (int y = st_y; y <= ed_y; ++y) {  
          for (int x = st_x; x <= ed_x; ++x) {

            int cnt = x+1 + (y+1)*3 + (z+1)*9;

            update_bound_send[cnt] = true;
            
            //get neigbhour proc coordincate.
            neighbour_procs[cnt][0] = coord[0] + x;
            neighbour_procs[cnt][1] = coord[1] + y;
            neighbour_procs[cnt][2] = coord[2] + z;

            //center block, not edge, does not need to update
            if (x == 0 && y == 0 && z == 0) {
              update_bound_send[cnt] = false;
              continue;
            }

            //if STENCIL_STAR, does not update corner blocks    
            int stencil_flag = abs(x) + abs(y) + abs(z);
            if (st == STENCIL_STAR && stencil_flag != 1) {
              update_bound_send[cnt] = false;
              continue;
            }

            // 
            if (xs == 0 && x == -1) {
              if (bx == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][0] = px - 1;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }
            
            if (xe == gx && x == 1) {
              if (bx == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][0] = 0;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }

            if (ys == 0 && y == -1) {
              if (by == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][1] = py - 1;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }

            if (ye == gy && y == 1) {
              if (by == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][1] = 0;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }
            
            if (zs == 0 && z == -1) {
              if (bz == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][2] = pz - 1;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }

            if (ze == gz && z == 1) {
              if (bz == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][2] = 0;
              } else {
                update_bound_send[cnt] = false;
                continue;
              }
            }

            // define the inside send box [starts, counts] of array_buffer
            int i_starts[3] = {i_cx[x + 1] + s, i_cy[y + 1] + s, i_cz[z + 1] + s};
            int i_counts[3] = {i_dx[x + 1], i_dy[y + 1], i_dz[z + 1]};
            i_box[cnt] = Box(i_starts, i_counts);

            // define the outside recieve box [starts, counts] of array_buffer
            int o_starts[3] = {o_cx[x + 1], o_cy[y + 1], o_cz[z + 1]};
            int o_counts[3] = {o_dx[x + 1], o_dy[y + 1], o_dz[z + 1]};
            o_box[cnt] = Box(o_starts, o_counts);


            update_cnt++;
          }
        }
      }

      int sw = s;

int c27=0;
      for (int i = 0; i < 27; ++i) {
        if (!update_bound_send[i]) continue;
        
        int xs1, ys1, zs1, xe1, ye1, ze1;
        i_box[i].get_corners(xs1, xe1, ys1, ye1, zs1, ze1);

        int ipx = neighbour_procs[i][0];
        int ipy = neighbour_procs[i][1];
        int ipz = neighbour_procs[i][2];

        int target_rank = pp->get_procs_rank(ipx, ipy, ipz);
        
        int starts[3]  = {xs1, ys1, zs1};
        int subsize[3] = {xe1-xs1, ye1-ys1, ze1-zs1};
        int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
        
        // debug
        // string str = "debug/" + to_string(pp->rank()) + ".out";
        // std::ofstream out(str, std::fstream::app);
        // auto coutbuf = cout.rdbuf(out.rdbuf());

        // vector<int> cd = pp->get_procs_3d(pp->rank());

        // cout<<"====Send====\n";
        // cout<<boost::format("from process [%d %d %d]\n") % cd[0] % cd[1] % cd[2];
        // cout<<boost::format("to process   [%d %d %d]\n") % ipx % ipy % ipz;
        // cout<<boost::format("starts  [%d %d %d]\n") % starts[0] % starts[1] % starts[2];
        // cout<<boost::format("bigsize [%d %d %d]\n") % bigsize[0] % bigsize[1] % bigsize[2];
        // cout<<boost::format("subsize [%d %d %d]\n") % subsize[0] % subsize[1] % subsize[2];
        // cout<<target_rank<<endl;
        // cout.rdbuf(coutbuf);        

        MPI_Datatype target_sub_array;
        MPI_Type_create_subarray(3, bigsize, subsize, starts,
                                 MPI_ORDER_FORTRAN,
                                 oa::utils::mpi_datatype(ap->get_data_type()),
                                 &target_sub_array);
        MPI_Type_commit(&target_sub_array);

        MPI_Request req;
        if(dev_type == CPU)
          MPI_Isend(ap->get_cpu_buffer(), 1, target_sub_array, target_rank, 100, pp->get_comm(), &req);
        else
          MPI_Isend(ap->get_buffer(), 1, target_sub_array, target_rank, 100, pp->get_comm(), &req);
        reqs.push_back(req);
        MPI_Type_free(&target_sub_array);
        c27++;
      }
        //printf("c27= %d \n",c27);
      // update_bound_recv
      if (direction != 4) {
        for (int i = 0; i < 27; i++) update_bound_recv[i] = update_bound_send[i];
      } else {
        st_x = -lb[0], ed_x = rb[0];
        st_y = -lb[1], ed_y = rb[1];
        st_z = -lb[2], ed_z = rb[2];

        for (int z = st_z; z <= ed_z; ++z) {
          for (int y = st_y; y <= ed_y; ++y) {  
            for (int x = st_x; x <= ed_x; ++x) {

              int cnt = x+1 + (y+1)*3 + (z+1)*9;

              update_bound_recv[cnt] = true;
              
              //get neigbhour proc coordincate.
              neighbour_procs[cnt][0] = coord[0] + x;
              neighbour_procs[cnt][1] = coord[1] + y;
              neighbour_procs[cnt][2] = coord[2] + z;

              //center block, not edge, does not need to update
              if (x == 0 && y == 0 && z == 0) {
                update_bound_recv[cnt] = false;
                continue;
              }

              //if STENCIL_STAR, does not update corner blocks    
              int stencil_flag = abs(x) + abs(y) + abs(z);
              if (st == STENCIL_STAR && stencil_flag != 1) {
                update_bound_recv[cnt] = false;
                continue;
              }

              // 
              if (xs == 0 && x == -1) {
                if (bx == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][0] = px - 1;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }
              
              if (xe == gx && x == 1) {
                if (bx == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][0] = 0;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }

              if (ys == 0 && y == -1) {
                if (by == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][1] = py - 1;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }

              if (ye == gy && y == 1) {
                if (by == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][1] = 0;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }
              
              if (zs == 0 && z == -1) {
                if (bz == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][2] = pz - 1;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }

              if (ze == gz && z == 1) {
                if (bz == BOUNDARY_PERIODIC) {
                  neighbour_procs[cnt][2] = 0;
                } else {
                  update_bound_recv[cnt] = false;
                  continue;
                }
              }

              // define the inside send box [starts, counts] of array_buffer
              int i_starts[3] = {i_cx[x + 1] + s, i_cy[y + 1] + s, i_cz[z + 1] + s};
              int i_counts[3] = {i_dx[x + 1], i_dy[y + 1], i_dz[z + 1]};
              i_box[cnt] = Box(i_starts, i_counts);

              // define the outside recieve box [starts, counts] of array_buffer
              int o_starts[3] = {o_cx[x + 1], o_cy[y + 1], o_cz[z + 1]};
              int o_counts[3] = {o_dx[x + 1], o_dy[y + 1], o_dz[z + 1]};
              o_box[cnt] = Box(o_starts, o_counts);


              update_cnt++;
            }
          }
        }
      }

      // receive should be reversed
      for (int i = 26; i >= 0; i--) {
      
        if (!update_bound_recv[i]) continue;
        
        int xs1, ys1, zs1, xe1, ye1, ze1;
        o_box[i].get_corners(xs1, xe1, ys1, ye1, zs1, ze1);

        int ipx = neighbour_procs[i][0];
        int ipy = neighbour_procs[i][1];
        int ipz = neighbour_procs[i][2];

        int target_rank = pp->get_procs_rank(ipx, ipy, ipz);
        
        int starts[3]  = {xs1, ys1, zs1};
        int subsize[3] = {xe1-xs1,ye1-ys1,ze1-zs1};
        int bigsize[3] = {xe-xs+2*sw,ye-ys+2*sw,ze-zs+2*sw};

        // debug
        // string str = "debug/" + to_string(pp->rank()) + ".out";
        // std::ofstream out(str, std::fstream::app);
        // auto coutbuf = cout.rdbuf(out.rdbuf());

        // vector<int> cd = pp->get_procs_3d(pp->rank());

        // cout<<"====Receive====\n";
        // cout<<boost::format("to process   [%d %d %d]\n") % cd[0] % cd[1] % cd[2];
        // cout<<boost::format("from process [%d %d %d]\n") % ipx % ipy % ipz;
        // cout<<boost::format("starts  [%d %d %d]\n") % starts[0] % starts[1] % starts[2];
        // cout<<boost::format("bigsize [%d %d %d]\n") % bigsize[0] % bigsize[1] % bigsize[2];
        // cout<<boost::format("subsize [%d %d %d]\n") % subsize[0] % subsize[1] % subsize[2];
        // cout<<target_rank<<endl;
        // cout.rdbuf(coutbuf);

        MPI_Datatype target_sub_array;
        MPI_Type_create_subarray(3, bigsize, subsize, starts,
                                 MPI_ORDER_FORTRAN,
                                 oa::utils::mpi_datatype(ap->get_data_type()),
                                 &target_sub_array);
        MPI_Type_commit(&target_sub_array);

        MPI_Request req;
        if(dev_type == CPU)
          MPI_Irecv(ap->get_cpu_buffer(), 1, target_sub_array, target_rank, 100, pp->get_comm(), &req);
        else
          MPI_Irecv(ap->get_buffer(), 1, target_sub_array, target_rank, 100, pp->get_comm(), &req);
        reqs.push_back(req);
        MPI_Type_free(&target_sub_array);
      }
    }

    void update_ghost_end(vector<MPI_Request> &reqs) {
      //cout<<reqs.size()<<endl;
      if (reqs.size() > 0) {
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
      }
      reqs.clear();
    }

    // set ghost to zeros, in order to check correctness
    void set_ghost_zeros(ArrayPtr ap) {
      void* buffer = ap->get_buffer();
      Shape ls = ap->local_shape();
      int sw = ap->get_stencil_width();

      #ifndef __HAVE_CUDA__ 
      switch (ap->get_data_type()) {
        case DATA_INT:
          oa::internal::set_ghost_consts((int*)buffer, ls, (int)0, sw);
          break;
        case DATA_FLOAT:
          oa::internal::set_ghost_consts((float*)buffer, ls, (float)0, sw);
          break;
        case DATA_DOUBLE:
          oa::internal::set_ghost_consts((double*)buffer, ls, (double)0, sw);
          break;
      }
      #else
      switch (ap->get_data_type()) {
        case DATA_INT:
          oa::gpu::set_ghost_consts((int*)buffer, ls, (int)0, sw);
          break;
        case DATA_FLOAT:
          oa::gpu::set_ghost_consts((float*)buffer, ls, (float)0, sw);
          break;
        case DATA_DOUBLE:
          oa::gpu::set_ghost_consts((double*)buffer, ls, (double)0, sw);
          break;
      }
      #endif
    }

    // set boundary to zeros, especially when use operator
    void set_boundary_zeros(ArrayPtr &ap, oa_int3 lb, oa_int3 rb) {
      Shape s = ap->shape();

      if (lb[0]) set_boundary_zeros(ap, Box(0, lb[0], 0, s[1], 0, s[2]));
      if (lb[1]) set_boundary_zeros(ap, Box(0, s[0], 0, lb[1], 0, s[2]));
      if (lb[2]) set_boundary_zeros(ap, Box(0, s[0], 0, s[1], 0, lb[2]));
      
      if (rb[0]) set_boundary_zeros(ap, Box(s[0] - rb[0], s[0], 0, s[1], 0, s[2]));
      if (rb[1]) set_boundary_zeros(ap, Box(0, s[0], s[1] - rb[1], s[1], 0, s[2]));
      if (rb[2]) set_boundary_zeros(ap, Box(0, s[0], 0, s[1], s[2] - rb[2], s[2]));
    }
    
    // set boundary to zeros, especially when use operator
    void set_boundary_zeros(ArrayPtr &ap, Box sub_box) {
      Shape s = ap->shape();
      switch(ap->get_data_type()) {
        case DATA_INT:
          set_ref_const(ap, sub_box, (int)0);
          break;
        case DATA_FLOAT:
          set_ref_const(ap, sub_box, (float)0);
          break;
        case DATA_DOUBLE:
          set_ref_const(ap, sub_box, (double)0);
          break;
      }
    }

#define calc_id(i,j,k,S) ((k)*(S[0])*(S[1])+(j)*(S[0])+(i))
/*
    inline int calc_id(int i, int j, int k, oa_int3 S) {
      int M = S[0];
      int N = S[1];
      int P = S[2];
      return k * M * N + j * M + i;
    }
*/
    // ap = Operator(A), only calc inside
    // lbound = [-xs_sw, -ys_sw, -zs_sw]
    // rbound = [xe_sw, ye_sw, ze_sw]
    void calc_inside(ArrayPtr &ap, ArrayPtr &A, oa_int3 lbound, oa_int3 rbound) {
      int sw = A->get_partition()->get_stencil_width();
      Shape sp = A->local_shape();
      Shape S = A->buffer_shape();

      int* b1 = (int*) ap->get_buffer();
      int* b2 = (int*) A->get_buffer();

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

    }

    void calc_outside(ArrayPtr &ap, ArrayPtr &A, oa_int3 lbound, oa_int3 rbound) {
      int sw = A->get_partition()->get_stencil_width();
      Shape sp = A->local_shape();
      Shape S = A->buffer_shape();

      int* b1 = (int*) ap->get_buffer();
      int* b2 = (int*) A->get_buffer(); 

      int M = S[0];
      int N = S[1];
      int P = S[2];

      // update outside six surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            b1[calc_id(i, j, k, S)] = b2[calc_id(i - 2, j - 1, k, S)] 
                + b2[calc_id(i + 1, j + 2, k + 1, S)];
          }
        }
      }
    }

    
    ArrayPtr to_rank0(ArrayPtr A){
      assert(A->get_partition() != NULL);
      
      if(A->is_seqs())
        return A;

      ArrayPtr B = zeros(MPI_COMM_SELF, A->shape(), 0, A->get_data_type());

      DataType dt = A->get_data_type();
      
      Shape ps = A->get_partition()->procs_shape();

      int npx = ps[0];
      int npy = ps[1];
      int npz = ps[2];
      
      int my_rank = A->rank();
        
      int num_procs = npx * npy * npz;

      Shape gs = A->shape();
      char* global_buf = (char*)B->get_buffer();
        
      MPI_Request reqs[num_procs];
      int reqs_cnt = 0;

      // rank 0 recv & others send
      if (my_rank == 0) {
        for(int z = 0; z < npz; ++z) {
          for(int y = 0; y < npy; ++y) {
            for(int x = 0; x < npx; ++x) {
                
              Box box = A->get_partition()->get_local_box({x, y, z});
              if (box.size() <= 0) continue;

              // int xs, ys, zs, xe, ye, ze;
              // box.get_corners(xs, xe, ys, ye, zs, ze);
                                        
              MPI_Datatype target_sub_array;
              int bigsize[3] = {gs[0], gs[1], gs[2]};
              
              MPI_Type_create_subarray(3, bigsize,
                                       box.counts().data(),
                                       box.starts().data(),
                                       MPI_ORDER_FORTRAN,
                                       oa::utils::mpi_datatype(dt),
                                       &target_sub_array);
                                        
              MPI_Type_commit(&target_sub_array);
              int target_rank = A->get_partition()->get_procs_rank({x, y, z});
              MPI_Irecv(global_buf, 1,
                        target_sub_array,
                        target_rank, 100,
                        A->get_partition()->get_comm(),
                        &reqs[reqs_cnt++]);
                                        
              MPI_Type_free(&target_sub_array);
            }
          }
        }
      }

      // all process send subarray (if size > 0) to global_buf
      Box box = A->get_local_box();
      if (box.size() > 0) {
        int xs, ys, zs, xe, ye, ze;
        box.get_corners(xs, xe, ys, ye, zs, ze);
        int sw = A->get_partition()->get_stencil_width();  
                
        MPI_Datatype mysubarray;
        int starts[3]  = {sw, sw, sw};
        int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
        int subsize[3] = {xe-xs, ye-ys, ze-zs};
        MPI_Type_create_subarray(3, bigsize, subsize,
                                 starts, MPI_ORDER_FORTRAN,
                                 oa::utils::mpi_datatype(dt),
                                 &mysubarray);
        MPI_Type_commit(&mysubarray);
        MPI_Send(A->get_buffer(), 1, mysubarray, 0, 100,
                 A->get_partition()->get_comm());
        MPI_Type_free(&mysubarray);
      }

      if (my_rank == 0){
        //m_par_ptr->display(NULL, true);
                
        MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);
        //oa::utils::print_data((void*)global_buf, gs, A->get_data_type());
        //delete(global_buf);
      }

      return B;
    }

    bool is_equal(const ArrayPtr& A, const ArrayPtr& B){

      if(!A->is_seqs()) return false;
      if(A->shape() != B->shape()) return false;
      int A_size = A->size();
      
      #ifndef __HAVE_CUDA__
      if(A->get_data_type() == DATA_INT){
        int* A_buf = (int*)A->get_buffer();
        int* B_buf = (int*)B->get_buffer();
        for(int i = 0; i < A_size; ++ i){
          if(abs(A_buf[i] - B_buf[i]) > 1E-8){
            return false;
          }
        }        
      }else if(A->get_data_type() == DATA_FLOAT){
        float* A_buf = (float*)A->get_buffer();
        float* B_buf = (float*)B->get_buffer();
        for(int i = 0; i < A_size; ++ i){
          if(fabs(A_buf[i] - B_buf[i]) > 1E-8){
            return false;
          }
        }        
      }else if(A->get_data_type() == DATA_DOUBLE){
        double* A_buf = (double*)A->get_buffer();
        double* B_buf = (double*)B->get_buffer();
        for(int i = 0; i < A_size; ++ i){
          if(fabs(A_buf[i] - B_buf[i]) > 1E-8){
            return false;
          }
        }        
      }
      return true;
      #else
      if(A->get_data_type() == DATA_INT)
        return oa::gpu::is_equal_array_and_array((int*)A->get_buffer(),(int*)B->get_buffer(), A_size);
      else if(A->get_data_type() == DATA_FLOAT)
        return oa::gpu::is_equal_array_and_array((float*)A->get_buffer(), (float*)B->get_buffer(), A_size);
      else if (A->get_data_type() == DATA_DOUBLE)
        return oa::gpu::is_equal_array_and_array((double*)A->get_buffer(),(double*)B->get_buffer(), A_size);
      return true;
      #endif
    }

    // sub(A) = A(A_box) and set sub(A) = B
    void set(ArrayPtr& A, const Box& sub_box,
            const ArrayPtr& B) {

      // need reset pseudo_3d
      A->reset_pseudo_3d();

      // sub(A)'shape must equal B's shape
      assert(B->shape() == sub_box.shape());
      // A->display("AAAA=");
      // B->display("BBBB=");
      // sub_box.display("sub_box=");
      
      // sub(A)'s partition
      vector<int> rsx, rsy, rsz;
      PartitionPtr pp = A->get_partition();
      // Shape ps = pp->procs_shape();
      // pp->split_box_procs(A_box, rsx, rsy, rsz);
      
      // vector<int> x(ps[0], 0), y(ps[1], 0), z(ps[2], 0);
      // for (int i = 0; i < rsx.size(); i += 3)
      //   x[rsx[i + 2]] = rsx[i + 1] - rsx[i];
      // for (int i = 0; i < rsy.size(); i += 3)
      //   y[rsy[i + 2]] = rsy[i + 1] - rsy[i];
      // for (int i = 0; i < rsz.size(); i += 3)
      //   z[rsz[i + 2]] = rsz[i + 1] - rsz[i];

      // PartitionPtr subA_par_ptr = PartitionPool::global()->
      //   get(pp->get_comm(), x, y, z, pp->get_stencil_width());

      PartitionPtr subA_par_ptr = pp->sub(sub_box);
      
      ArrayPtr ap;
      // if sub(A)'s partition doesn't equal to B's partition,
      // needs transfer
      if (!subA_par_ptr->equal(B->get_partition())) {
        ap = transfer(B, subA_par_ptr);
      }else{
        ap = B;
      }
      // ap->display("CCC=");

      // don't have local data in process
      if (!ap->has_local_data()) return ;

      // int rk = pp->rank();
      // vector<int> procs_coord = pp->get_procs_3d(rk);

      // int idx = procs_coord[0] - rsx[2];
      // int idy = procs_coord[1] - rsy[2];
      // int idz = procs_coord[2] - rsz[2];

      Box A_local_box = A->get_local_box();
      Box A_sub_box = A_local_box.get_intersection(sub_box)
        .ref_box(A_local_box);
      Shape s = ap->local_shape();
      Box B_sub_box(0, s[0], 0, s[1], 0, s[2]);
      
      // Box sub_box(
      //             rsx[idx * 3], rsx[idx * 3 + 1],
      //             rsy[idy * 3], rsy[idy * 3 + 1], 
      //             rsz[idz * 3], rsz[idz * 3 + 1]
      //             );

      //sub_box.display("sub_box = ");

      int sw = A->get_stencil_width();
      
      // A_sub_box.display("A_sub_box = ");
      // B_sub_box.display("B_sub_box = ");

      // A_sub_box.shift(sw).display("A_sub_box1 = ");
      // B_sub_box.shift(sw).display("B_sub_box1 = ");
      
      // different data_type
      
      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_INT) {
        oa::internal::copy_buffer<int, int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_INT) {
        oa::gpu::copy_buffer<int, int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_FLOAT) {
        oa::internal::copy_buffer<int, float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_FLOAT) {
        oa::gpu::copy_buffer<int, float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::internal::copy_buffer<int, double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_INT
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::gpu::copy_buffer<int, double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_INT) {
        oa::internal::copy_buffer<float, int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_INT) {
        oa::gpu::copy_buffer<float, int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_FLOAT) {
        oa::internal::copy_buffer<float, float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_FLOAT) {
        oa::gpu::copy_buffer<float, float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::internal::copy_buffer<float, double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_FLOAT
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::gpu::copy_buffer<float, double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_INT) {
        oa::internal::copy_buffer<double, int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_INT) {
        oa::gpu::copy_buffer<double, int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (int*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_FLOAT) {
        oa::internal::copy_buffer<double, float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_FLOAT) {
        oa::gpu::copy_buffer<double, float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (float*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

      #ifndef __HAVE_CUDA__
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::internal::copy_buffer<double, double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #else 
      if (A->get_data_type() == DATA_DOUBLE
              && ap->get_data_type() == DATA_DOUBLE) {
        oa::gpu::copy_buffer<double, double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A_sub_box.shift(sw),
          (double*) ap->get_buffer(),
          ap->buffer_shape(),
          B_sub_box.shift(sw)
        );
      }
      #endif

    
      // A->display("AAAA1111=");
    }

    // sub(A) = A(A_box), sub(B) = B(B_box) && set sub(A) = sub(B)
    void set(ArrayPtr& A, const Box& A_box, 
        const ArrayPtr& B, const Box& B_box) {
      
      // make sure sub(A).shape == sub(B).shape
      assert(A_box.shape() == B_box.shape());

      ArrayPtr subB = subarray(B, B_box);

      set(A, A_box, subB);

    }
      
    ArrayPtr l2g(ArrayPtr& lap)
    {
      #ifndef __HAVE_CUDA__
      ArrayPtr gap;

      PartitionPtr lpp = lap->get_partition();

      Shape lsp = lpp->shape();
      Shape gsp = lsp;
      int sw = lpp->get_stencil_width(); 
      int datatype = lap->get_data_type();
      gap = zeros(oa::MPI::global()->comm(), gsp, sw, datatype);
      PartitionPtr gpp = gap->get_partition();
      Box gbox = gpp->get_local_box();
      int xs, ys, zs, xe, ye, ze;
      gbox.get_corners(xs, xe, ys, ye, zs, ze);

      int lxs, lys, lzs, lxe, lye, lze;
      Box lbox =lpp->get_local_box();
      lbox.get_corners(lxs, lxe, lys, lye, lzs, lze);

      vector<int> clx = gpp->m_clx;
      vector<int> cly = gpp->m_cly;
      vector<int> clz = gpp->m_clz;

      // int mpisize = oa::utils::get_size(gpp->get_comm());
      // int myrank = oa::utils::get_rank(gpp->get_comm());

      int mpisize = MPI_SIZE;
      int myrank  = MPI_RANK;

      vector<int> location = gpp->get_procs_3d(myrank);

      switch(datatype) {
        case DATA_INT:
          {
            int *gbuff = (int *) gap->get_buffer();
            int *lbuff = (int *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
              for (int j = ys; j < ye; j++) {
                for (int i = xs; i < xe; i++) {
                  int gindex = (xe-xs+2*sw)*(ye-ys+2*sw)*(k-zs+sw)+(xe-xs+2*sw)*(j-ys+sw)+(i-xs+sw);
                  gbuff[gindex] = 1;
                  int lk = k + clx[location[0]];
                  int lj = j + cly[location[1]];
                  int li = i + clz[location[2]];
                  int lindex = (lxe-lxs+2*sw)*(lye-lys+2*sw)*(k+sw)+(lxe-lxs+2*sw)*(j+sw)+(i+sw);
                  gbuff[gindex] = lbuff[lindex];
                }
              }
            }
            break;
          }
        case DATA_FLOAT:
          {
            float *gbuff = (float *) gap->get_buffer();
            float *lbuff = (float *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
              for (int j = ys; j < ye; j++) {
                for (int i = xs; i < xe; i++) {
                  int gindex = (xe-xs+2*sw)*(ye-ys+2*sw)*(k-zs+sw)+(xe-xs+2*sw)*(j-ys+sw)+(i-xs+sw);
                  gbuff[gindex] = 1;
                  int lk = k + clx[location[0]];
                  int lj = j + cly[location[1]];
                  int li = i + clz[location[2]];
                  int lindex = (lxe-lxs+2*sw)*(lye-lys+2*sw)*(k+sw)+(lxe-lxs+2*sw)*(j+sw)+(i+sw);
                  gbuff[gindex] = lbuff[lindex];
                }
              }
            }
            break;
          }
        case DATA_DOUBLE:
          {
            double *gbuff = (double *) gap->get_buffer();
            double *lbuff = (double *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
              for (int j = ys; j < ye; j++) {
                for (int i = xs; i < xe; i++) {
                  int gindex = (xe-xs+2*sw)*(ye-ys+2*sw)*(k-zs+sw)+(xe-xs+2*sw)*(j-ys+sw)+(i-xs+sw);
                  gbuff[gindex] = 1;
                  int lk = k + clx[location[0]];
                  int lj = j + cly[location[1]];
                  int li = i + clz[location[2]];
                  int lindex = (lxe-lxs+2*sw)*(lye-lys+2*sw)*(k+sw)+(lxe-lxs+2*sw)*(j+sw)+(i+sw);
                  gbuff[gindex] = lbuff[lindex];
                }
              }
            }
            break;
          }
        default:
          std::cout<<"err"<<std::endl;
          break;
      }

      return gap;
      #else 
      return oa::gpu::l2g_gpu(lap);
      #endif
    }

    ArrayPtr g2l(ArrayPtr& gap)
    {
      PartitionPtr m_par_ptr = gap->get_partition();
      Shape ps = m_par_ptr->procs_shape();
      int m_data_type = gap->get_data_type();
      int npx = ps[0];
      int npy = ps[1];
      int npz = ps[2];

      MPI_Comm comm = m_par_ptr->get_comm();
      //int my_rank = oa::utils::get_rank(comm);
      int my_rank = oa::MPI::global()->rank(comm);

      int num_procs = npx * npy * npz;

      Shape gs = gap->shape();

      MPI_Request reqs[num_procs];
      int reqs_cnt = 0;
      ArrayPtr lap = zeros(MPI_COMM_SELF, gs, 0, m_data_type);
      void * global_buf = lap->get_buffer();
      // rank 0 recv & others send
      if (my_rank == 0) {
        for(int z = 0; z < npz; ++z) {
          for(int y = 0; y < npy; ++y) {
            for(int x = 0; x < npx; ++x) {

              Box box = m_par_ptr -> get_local_box({x, y, z});
              if (box.size() <= 0) continue;

              int xs, ys, zs, xe, ye, ze;
              box.get_corners(xs, xe, ys, ye, zs, ze);

              MPI_Datatype target_sub_array;
              int starts[3] = {xs, ys, zs};
              int subsize[3] = {xe-xs, ye-ys, ze-zs};
              int bigsize[3] = {gs[0], gs[1], gs[2]};

              MPI_Type_create_subarray(3, bigsize, subsize,
                  starts, MPI_ORDER_FORTRAN,
                  oa::utils::mpi_datatype(m_data_type),
                  &target_sub_array);

              MPI_Type_commit(&target_sub_array);
              int target_rank = m_par_ptr->get_procs_rank({x, y, z});
              MPI_Irecv(global_buf, 1,
                  target_sub_array,
                  target_rank, 100,
                  m_par_ptr->get_comm(),
                  &reqs[reqs_cnt++]);

              MPI_Type_free(&target_sub_array);
            }
          }
        }

      }

      // all process send subarray (if size > 0) to global_buf
      Box box = m_par_ptr -> get_local_box();
      if (box.size() > 0) {
        int xs, ys, zs, xe, ye, ze;
        box.get_corners(xs, xe, ys, ye, zs, ze);
        int sw = m_par_ptr -> get_stencil_width();  

        MPI_Datatype mysubarray;
        int starts[3]  = {sw, sw, sw};
        int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
        int subsize[3] = {xe-xs, ye-ys, ze-zs};
        MPI_Type_create_subarray(3, bigsize, subsize,
            starts, MPI_ORDER_FORTRAN,
            oa::utils::mpi_datatype(m_data_type),
            &mysubarray);
        MPI_Type_commit(&mysubarray);
        MPI_Send(gap->get_buffer(), 1, mysubarray, 0, 100, m_par_ptr->get_comm());
        MPI_Type_free(&mysubarray);
      }
      MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);
      MPI_Bcast(global_buf, gs[0]*gs[1]*gs[2], oa::utils::mpi_datatype(m_data_type), 0, comm);  
      return lap;
    }

    // sub(A) = B (MPI_COMM_SELF)
    void set_l2g(ArrayPtr& A, const Box& A_box, ArrayPtr& B) {
      ArrayPtr gap = l2g(B);
      set(A, A_box, gap);
    }

    // local_A (MPI_COMM_SELF)= sub(global_B)
    void set_g2l(ArrayPtr& local, const Box& sub_box, ArrayPtr& global) {
      ArrayPtr sub = subarray(global, sub_box);
      local = g2l(sub);
    }

    ArrayPtr make_psudo3d(const ArrayPtr& B){
      Shape ps = B->get_partition()->procs_shape();
      Shape as = B->shape();

      for (int i = 0; i < 3; i++) {
        if (as[i] != 1) {
          ps[i] = 1;
        }
      }

      if (ps[0] == 1 && ps[1] == 1 && ps[2] == 1) {
        // printf("why????????\n");
        // return B;
      }
      
      // if(ps[0] == 1 && ps[1] == 1 && ps[2] == 1){
      //   THROW_LOGIC_EXCEPTION(
      //       "cannot make a psudo 3d array for a real 3d array.");
      // }
      
      NodePtr n3 =
        NodePool::global()->get_local_1d<int, 3>(ps.data());

      std::vector<ArrayPtr> apl;
      apl.push_back(B);
      apl.push_back(n3->get_data());

      ArrayPtr ap =
        oa::kernel::kernel_rep_with_partition(apl, true);
      
      ap->set_bitset(B->get_bitset());
      ap->set_pseudo(false);
      ap->set_pos(B->get_pos());
      return ap;
    }

        void set_with_mask(ArrayPtr& A, 
                       const ArrayPtr& B, 
                       const ArrayPtr& M) {
      // A->display("A");
      // B->display("B");
      // M->display("M");      
      // need reset pseudo_3d
      A->reset_pseudo_3d();

      bool ifscalar_B = B->is_seqs_scalar();
      if(!ifscalar_B){
        assert(B->shape() == A->shape());
      }
      
      assert(M->shape() == A->shape());
      
      ArrayPtr B1, M1;
      
      // needs transfer
      if (!A->get_partition()->equal(
                  B->get_partition()) && !ifscalar_B) {
        B1 = transfer(B, A->get_partition());
      }else{
        B1 = B;
      }
      
      // needs transfer
      if (!A->get_partition()->equal(M->get_partition()) ) {
        M1 = transfer(M, A->get_partition());
      }else{
        M1 = M;
      }

      // don't have local data in process
      if (!A->has_local_data())
        return ;

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            int,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            int,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            int,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            int,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            int,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            int,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            float,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            float,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            float,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            float,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            float,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            float,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            double,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            double,
                                            int>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            double,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            double,
                                            float>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_INT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<int,
                                            double,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<int,
                                            double,
                                            double>(
          (int*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            int,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            int,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            int,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            int,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            int,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            int,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            float,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            float,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            float,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            float,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            float,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            float,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            double,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            double,
                                            int>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            double,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            double,
                                            float>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_FLOAT
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<float,
                                            double,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<float,
                                            double,
                                            double>(
          (float*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            int,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            int,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            int,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            int,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_INT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            int,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            int,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (int*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            float,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            float,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            float,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            float,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_FLOAT 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            float,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            float,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (float*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_INT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            double,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            double,
                                            int>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (int*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_FLOAT ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            double,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            double,
                                            float>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (float*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

      if (A->get_data_type() == DATA_DOUBLE
          && B1->get_data_type() == DATA_DOUBLE 
          && M1->get_data_type() == DATA_DOUBLE ) {
        #ifndef __HAVE_CUDA__
        oa::internal::copy_buffer_with_mask<double,
                                            double,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #else
        oa::gpu::copy_buffer_with_mask<double,
                                            double,
                                            double>(
          (double*) A->get_buffer(),
          A->buffer_shape(),
          A->local_data_win(),
          (double*) B1->get_buffer(),
          B1->buffer_shape(),
          B1->local_data_win(),
          (double*) M1->get_buffer(),
          M1->buffer_shape(),
          M1->local_data_win(),
          ifscalar_B
        );
        #endif
        return;
      }

        
    }

  }
}
