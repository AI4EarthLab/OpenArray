#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "utils/utils.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"

namespace oa {
	namespace funcs {

		//create an array with const val
		template <typename T>
		ArrayPtr consts(MPI_Comm comm, const Shape& s, T val, int stencil_width = 1) {
			int data_type = oa::utils::to_type<T>();
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
			Box box = ap->get_local_box();
			int size = box.size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

		//create an array with const val
		template <typename T>
		ArrayPtr consts(MPI_Comm comm, const vector<int>& x, const vector<int>& y, 
			const vector<int>&z, T val, int stencil_width = 1) {
			int data_type = oa::utils::to_type<T>();
			ArrayPtr ap = ArrayPool::global()->get(comm, x, y, z, stencil_width, data_type);
			Box box = ap->get_local_box();
			int size = box.size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

		// create a ones array
		ArrayPtr ones(MPI_Comm comm, const Shape& s, 
			int stencil_width = 1, int data_type = DATA_INT);

		// create a zeros array
		ArrayPtr zeros(MPI_Comm comm, const Shape& s, 
			int stencil_width = 1, int data_type = DATA_INT);

		// create a rand array
		ArrayPtr rand(MPI_Comm comm, const Shape& s, 
			int stencil_width = 1, int data_type = DATA_INT);

		// create a seqs array
		ArrayPtr seqs(MPI_Comm comm, const Shape& s, 
			int stencil_width = 1, int data_type = DATA_INT); 

		ArrayPtr seqs(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
      const vector<int> &z, int stencil_width = 1, int data_type = DATA_INT);

		// according to partiton pp, transfer src to A
		// A = transfer(src, pp)
		ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp);

		// get a sub Array based on Box b
		ArrayPtr subarray(const ArrayPtr &ap, const Box &box);

		/*
     * update boundary, direction = -1, all dimension
     * direction = 0, dimension x
     * direction = 1, dimension y
     * direction = 2, dimension z
     */
    void update_ghost_start(ArrayPtr ap, vector<MPI_Request> &reqs, int direction = -1);
		
		void update_ghost_end(vector<MPI_Request> &reqs);
		
	}
}

#endif
