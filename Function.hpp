#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "utils/to_type.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"

namespace oa {
	namespace funcs {
		template <typename T>
		//create an array with const val
		ArrayPtr consts(MPI_Comm comm, const Shape& s, T val, int stencil_width = 1) {
			int data_type = oa::utils::to_type<T>();
			ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
			BoxPtr box_ptr = ap->get_corners();
			int size = box_ptr->size(stencil_width);
			oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
			return ap;
		}

		template <typename T>
		ArrayPtr consts(MPI_Comm comm, const vector<int>& x, const vector<int>& y, 
			const vector<int>&z, int stencil_width, T val) {

		}
		//if (boost::is_type<T, int>::value())
		ArrayPtr consts(int m = 1, int n = 1, int p = 1, float x = 1.0);
		ArrayPtr consts(int m = 1, int n = 1, int p = 1, double x = 1.0);
		ArrayPtr ones(int m = 1, int n = 1, int p = 1);
		ArrayPtr zeros(int m = 1, int n = 1, int p = 1);
		ArrayPtr rand(int m = 1, int n = 1, int p = 1);
		ArrayPtr seqs(int m = 1, int n = 1, int p = 1);
		//transfer(ArrayPtr &A, ArrayPtr &B);
	}
}

#endif
