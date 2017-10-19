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
			Box box = ap->get_corners();
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
			Box box = ap->get_corners();
			int size = box.size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

		// create a ones array
		ArrayPtr ones(MPI_Comm comm, const Shape& s, int stencil_width);

		// create a zeros array
		ArrayPtr zeros(MPI_Comm comm, const Shape& s, int stencil_width);

		// create a rand array
		ArrayPtr rand(MPI_Comm comm, const Shape& s, int stencil_width);

		// create a seqs array
		ArrayPtr seqs(MPI_Comm comm, const Shape& s, int stencil_width); 

		// according to partiton pp, transfer B to A
		ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp);

		// get a sub Array based on Box b
		ArrayPtr subarray(const ArrayPtr &ap, const Box &box);
	}
}

#endif
