#include "Function.hpp"

namespace oa {
	namespace funcs {

		// create a ones array
		ArrayPtr ones(MPI_Comm comm, const Shape& s, int stencil_width = 1) {
			ArrayPtr ap = consts(comm, s, 1, stencil_width);
			return ap;
		}

		// create a zeros array
		ArrayPtr zeros(MPI_Comm comm, const Shape& s, int stencil_width = 1) {
			ArrayPtr ap = consts(comm, s, 0, stencil_width);
			return ap;
		}

		// create a rand array
		ArrayPtr rand(MPI_Comm comm, const Shape& s, int stencil_width = 1) {
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, DATA_INT);
			Box box = ap->get_corners();
			int size = box.size(stencil_width);
			oa::internal::set_buffer_rand((int*)ap->get_buffer(), size);
			return ap;
		}

		// create a seqs array
		ArrayPtr seqs(MPI_Comm comm, const Shape& s, int stencil_width = 1) {
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, DATA_INT);
			Box box = ap->get_corners();
			oa::internal::set_buffer_seqs((int*)ap->get_buffer(), s, box, stencil_width);
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

  		if (!arr_ptr->has_local_data()) return arr_ptr; // don't have local data in the process
  		
  		int rk = pp->rank();
  		vector<int> procs_coord = pp->get_procs_3d(rk);

  		int idx = procs_coord[0] - rsx[2];
  		int idy = procs_coord[1] - rsy[2];
  		int idz = procs_coord[2] - rsz[2];

  		Box box = ap->get_corners();
  		Box sub_box(
  			rsx[idx * 3], rsx[idx * 3 + 1] - 1,
  			rsy[idy * 3], rsy[idy * 3 + 1] - 1, 
  			rsz[idz * 3], rsz[idz * 3 + 1] - 1
  		);

  		// different data_type
  		switch(ap->get_data_type()) {
  			case DATA_INT:
  				oa::internal::set_buffer_subarray<int>(
  					(int*) arr_ptr->get_buffer(), (int*) ap->get_buffer(), 
  					sub_box, box, pp->get_stencil_width()
  				);
  				break;
  			case DATA_FLOAT:
  				oa::internal::set_buffer_subarray<float>(
  					(float*) arr_ptr->get_buffer(), (float*) ap->get_buffer(), 
  					sub_box, box, pp->get_stencil_width()
  				);
  				break;
  			case DATA_DOUBLE:
  				oa::internal::set_buffer_subarray<double>(
  					(double*) arr_ptr->get_buffer(), (double*) ap->get_buffer(), 
  					sub_box, box, pp->get_stencil_width()
  				);
  				break;
  		}
  		
  		return arr_ptr;
		}

    ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp) {
      ArrayPtr ap = ArrayPool::global()->get(pp, src->get_data_type());

      int sw = pp->get_stencil_width();

      // src has local data, transfer to ap
      if (src->has_local_data()) {
        vector<int> rsx, rsy, rsz;
        Box src_box = src->get_corners();
        pp->split_box_procs(src_box, rsx, rsy, rsz);

        int xs, ys, zs, xe, ye, ze;
        src_box.get_corners(xs, xe, ys, ye, zs, ze);

        for (int i = 0; i < rsx.size(); i += 3) {
          for (int j = 0; j < rsy.size(); j += 3) {
            for (int k = 0; k < rsz.size(); k += 3) {
              MPI_Datatype src_subarray;
              int starts[3]  = {sw + rsx[i], sw + rsy[j], sw + rsz[k]};
              int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
              int subsize[3] = {rsx[i+1] - rsx[i], rsy[i+1] - rsy[i], rsz[i+1] - rsz[i]};
              MPI_Type_create_subarray(3, bigsize, subsize,
                starts, MPI_ORDER_FORTRAN,
                oa::utils::mpi_datatype(src->get_data_type()),
                &src_subarray);
              MPI_Type_commit(&src_subarray);

              int target_rank = pp->
                get_procs_rank({rsx[i + 2], rsy[i + 2], rsz[i + 2]});
              MPI_Send(src->get_buffer(), 
                1, src_subarray, target_rank, 100, pp->get_comm());
              MPI_Type_free(&src_subarray);
            }
          }
        }
      }

      MPI_Request reqs[pp->procs_size()];
      int reqs_cnt = 0;

      // ap has local data, receive from other processes
      if (ap->has_local_data()) {
        vector<int> rsx, rsy, rsz;
        Box dest_box = ap->get_corners();
        src->get_partition()->split_box_procs(dest_box, rsx, rsy, rsz);

        int xs, ys, zs, xe, ye, ze;
        dest_box.get_corners(xs, xe, ys, ye, zs, ze);

        for (int i = 0; i < rsx.size(); i += 3) {
          for (int j = 0; j < rsy.size(); j += 3) {
            for (int k = 0; k < rsz.size(); k += 3) {
              MPI_Datatype dest_subarray;
              int starts[3]  = {sw + rsx[i], sw + rsy[j], sw + rsz[k]};
              int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
              int subsize[3] = {rsx[i+1] - rsx[i], rsy[i+1] - rsy[i], rsz[i+1] - rsz[i]};
              MPI_Type_create_subarray(3, bigsize, subsize,
                starts, MPI_ORDER_FORTRAN,
                oa::utils::mpi_datatype(ap->get_data_type()),
                &dest_subarray);
              MPI_Type_commit(&dest_subarray);

              int target_rank = src->get_partition()->
                get_procs_rank({rsx[i + 2], rsy[i + 2], rsz[i + 2]});
              MPI_Irecv(ap->get_buffer(), 1,
                dest_subarray, target_rank, 
                100, pp->get_comm(),
                &reqs[reqs_cnt++]);

              MPI_Type_free(&dest_subarray);
            }
          }
        } 
      }

      MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);

      return ap;
    }

	}
}