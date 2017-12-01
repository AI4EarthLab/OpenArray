#include "Function.hpp"
#include "common.hpp"
#include "utils/utils.hpp"
#include <fstream>
#include <mpi.h>

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
    ArrayPtr rand(MPI_Comm comm, const Shape& s, 
                  int stencil_width, int data_type) {
      ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size(stencil_width);
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
                  int stencil_width, int data_type) {
      ArrayPtr ap = ArrayPool::global()->
        get(comm, s, stencil_width, data_type);
      Box box = ap->get_local_box();
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
      return ap;
    }

    ArrayPtr seqs(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
                  const vector<int> &z, int stencil_width, int data_type) {
      ArrayPtr ap = ArrayPool::global()->
        get(comm, x, y, z, stencil_width, data_type);
      Box box = ap->get_local_box();
      Shape s = ap->shape();
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
      return ap;
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
      
      ArrayPtr arr_ptr = ArrayPool::global()->
        get(pp->get_comm(), x, y, z, pp->get_stencil_width(), ap->get_data_type());

      if (!arr_ptr->has_local_data()) return arr_ptr; // don't have local data in process
      
      int rk = pp->rank();
      vector<int> procs_coord = pp->get_procs_3d(rk);

      int idx = procs_coord[0] - rsx[2];
      int idy = procs_coord[1] - rsy[2];
      int idz = procs_coord[2] - rsz[2];

      Box box = ap->get_local_box();
      Box sub_box(
                  rsx[idx * 3], rsx[idx * 3 + 1] - 1,
                  rsy[idy * 3], rsy[idy * 3 + 1] - 1, 
                  rsz[idz * 3], rsz[idz * 3 + 1] - 1
                  );

      // different data_type
      switch(ap->get_data_type()) {
      case DATA_INT:
        oa::internal::get_buffer_subarray<int>((int*) arr_ptr->get_buffer(),
                                               (int*) ap->get_buffer(),
                                               sub_box, box,
                                               pp->get_stencil_width());
        break;
      case DATA_FLOAT:
        oa::internal::get_buffer_subarray<float>((float*) arr_ptr->get_buffer(),
                                                 (float*) ap->get_buffer(),
                                                 sub_box, box, pp->get_stencil_width());
        break;
      case DATA_DOUBLE:
        oa::internal::get_buffer_subarray<double>((double*) arr_ptr->get_buffer(),
                                                  (double*) ap->get_buffer(), 
                                                  sub_box, box, pp->get_stencil_width());
        break;
      }
      return arr_ptr;
    }
    
    //transfer src to dest based on dest's partition pp
    ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp) {
      ArrayPtr ap = ArrayPool::global()->get(pp, src->get_data_type());

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

    void update_ghost_start(ArrayPtr ap, vector<MPI_Request> &reqs, int direction) {

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
      bool update_bound[27] = {false};

      Box i_box[27], o_box[27];

      int update_cnt = 0;

      if (!ap->has_local_data()) return ;

      int st_x, st_y, st_z, ed_x, ed_y, ed_z;
      st_x = st_y = st_z = ed_x = ed_y = ed_z = 0;

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
      }

      for (int z = st_z; z <= ed_z; ++z) {
        for (int y = st_y; y <= ed_y; ++y) {  
          for (int x = st_x; x <= ed_x; ++x) {

            int cnt = x+1 + (y+1)*3 + (z+1)*9;

            update_bound[cnt] = true;
            
            //get neigbhour proc coordincate.
            neighbour_procs[cnt][0] = coord[0] + x;
            neighbour_procs[cnt][1] = coord[1] + y;
            neighbour_procs[cnt][2] = coord[2] + z;

            //center block, not edge, does not need to update
            if (x == 0 && y == 0 && z == 0) {
              update_bound[cnt] = false;
              continue;
            }

            //if STENCIL_STAR, does not update corner blocks    
            int stencil_flag = abs(x) + abs(y) + abs(z);
            if (st == STENCIL_STAR && stencil_flag != 1) {
              update_bound[cnt] = false;
              continue;
            }

            // 
            if (xs == 0 && x == -1) {
              if (bx == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][0] = px - 1;
              } else {
                update_bound[cnt] = false;
                continue;
              }
            }
            
            if (xe == gx && x == 1) {
              if (bx == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][0] = 0;
              } else {
                update_bound[cnt] = false;
                continue;
              }
            }

            if (ys == 0 && y == -1) {
              if (by == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][1] = py - 1;
              } else {
                update_bound[cnt] = false;
                continue;
              }
            }

            if (ye == gy && y == 1) {
              if (by == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][1] = 0;
              } else {
                update_bound[cnt] = false;
                continue;
              }
            }
            
            if (zs == 0 && z == -1) {
              if (bz == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][2] = pz - 1;
              } else {
                update_bound[cnt] = false;
                continue;
              }
            }

            if (ze == gz && z == 1) {
              if (bz == BOUNDARY_PERIODIC) {
                neighbour_procs[cnt][2] = 0;
              } else {
                update_bound[cnt] = false;
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

      for (int i = 0; i < 27; ++i) {
        if (!update_bound[i]) continue;
        
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
        MPI_Isend(ap->get_buffer(), 1, target_sub_array, target_rank, 100,
                  pp->get_comm(), &req);
        reqs.push_back(req);
        MPI_Type_free(&target_sub_array);
      }

      for (int i = 0; i < 27; ++i) {
        if (!update_bound[i]) continue;
        
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
        MPI_Irecv(ap->get_buffer(), 1, target_sub_array, target_rank, 100,
                  pp->get_comm(), &req);
        reqs.push_back(req);
        MPI_Type_free(&target_sub_array);
      }
    }

    void update_ghost_end(vector<MPI_Request> &reqs) {
      //cout<<reqs.size()<<endl;
      if (reqs.size() > 0) {
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
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
          if(abs(A_buf[i] - B_buf[i]) > 1E-8){
            return false;
          }
        }        
      }else if(A->get_data_type() == DATA_DOUBLE){
        double* A_buf = (double*)A->get_buffer();
        double* B_buf = (double*)B->get_buffer();
        for(int i = 0; i < A_size; ++ i){
          if(abs(A_buf[i] - B_buf[i]) > 1E-8){
            return false;
          }
        }        
      }
      

      return true;
    }

    // sub(A) = A(A_box) and set sub(A) = B
    void set(ArrayPtr& A, const Box& A_box, ArrayPtr& B) {
      // sub(A)'shape must equal B's shape
      assert(B->shape() == A_box.shape());

      // sub(A)'s partition
      vector<int> rsx, rsy, rsz;
      PartitionPtr pp = A->get_partition();
      Shape ps = pp->procs_shape();
      pp->split_box_procs(A_box, rsx, rsy, rsz);
      
      vector<int> x(ps[0], 0), y(ps[1], 0), z(ps[2], 0);
      for (int i = 0; i < rsx.size(); i += 3)
        x[rsx[i + 2]] = rsx[i + 1] - rsx[i];
      for (int i = 0; i < rsy.size(); i += 3)
        y[rsy[i + 2]] = rsy[i + 1] - rsy[i];
      for (int i = 0; i < rsz.size(); i += 3)
        z[rsz[i + 2]] = rsz[i + 1] - rsz[i];

      PartitionPtr subA_par_ptr = PartitionPool::global()->
        get(pp->get_comm(), x, y, z, pp->get_stencil_width());

      ArrayPtr ap = B;
      // if sub(A)'s partition doesn't equal to B's partition, needs transfer
      if (!subA_par_ptr->equal(B->get_partition())) {
        ap = transfer(B, subA_par_ptr);
      }

      // don't have local data in process
      if (!ap->has_local_data()) return ;

      int rk = pp->rank();
      vector<int> procs_coord = pp->get_procs_3d(rk);

      int idx = procs_coord[0] - rsx[2];
      int idy = procs_coord[1] - rsy[2];
      int idz = procs_coord[2] - rsz[2];

      Box box = A->get_local_box();
      Box sub_box(
                  rsx[idx * 3], rsx[idx * 3 + 1] - 1,
                  rsy[idy * 3], rsy[idy * 3 + 1] - 1, 
                  rsz[idz * 3], rsz[idz * 3 + 1] - 1
                  );

      // different data_type

      ///:set TYPE = [['DATA_INT', 'int'], ['DATA_FLOAT', 'float'], ['DATA_DOUBLE', 'double']]
      ///:for i in TYPE
      ///:for j in TYPE
      if (A->get_data_type() == ${i[0]}$ && ap->get_data_type() == ${j[0]}$) {
        oa::internal::set_buffer_subarray<${i[1]}$, ${j[1]}$>(
          (${i[1]}$*) A->get_buffer(),
          (${j[1]}$*) ap->get_buffer(),
          box,
          sub_box,
          pp->get_stencil_width()
        );
      }

      ///:endfor
      ///:endfor

    }
    
    // sub(A) = A(A_box), sub(B) = B(B_box) && set sub(A) = sub(B)
    void set(ArrayPtr& A, const Box& A_box, 
        const ArrayPtr& B, const Box& B_box) {
      
      // make sure sub(A).shape == sub(B).shape
      assert(A_box.shape() == B_box.shape());

      ArrayPtr subB = subarray(B, B_box);
      set(A, A_box, subB);

    }
  }
}
