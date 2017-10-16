#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "utils/to_type.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"

namespace oa {
	namespace funcs {

		//create an array with const val
		template <typename T>
		ArrayPtr consts(MPI_Comm comm, const Shape& s, T val, int stencil_width = 1) {
			int data_type = oa::utils::to_type<T>();
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
			BoxPtr box_ptr = ap->get_corners();
			int size = box_ptr->size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

		//create an array with const val
		template <typename T>
		ArrayPtr consts(MPI_Comm comm, const vector<int>& x, const vector<int>& y, 
			const vector<int>&z, T val, int stencil_width = 1) {
			int data_type = oa::utils::to_type<T>();
			ArrayPtr ap = ArrayPool::global()->get(comm, x, y, z, stencil_width, data_type);
			BoxPtr box_ptr = ap->get_corners();
			int size = box_ptr->size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

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
			BoxPtr box_ptr = ap->get_corners();
			int size = box_ptr->size(stencil_width);
			oa::internal::set_buffer_rand((int*)ap->get_buffer(), size);
			return ap;
		}

		// create a seqs array
		ArrayPtr seqs(MPI_Comm comm, const Shape& s, int stencil_width = 1) {
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, DATA_INT);
			BoxPtr box_ptr = ap->get_corners();
			oa::internal::set_buffer_seqs((int*)ap->get_buffer(), box_ptr, stencil_width);
		}
		//transfer(ArrayPtr &A, ArrayPtr &B);
	}
}

#endif
