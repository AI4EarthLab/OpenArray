#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "ArrayPool.hpp"

namespace oa {
	namespace funcs {
		tempalte<typename T>
		ArrayPtr consts(MPI_COMM comm, const Shape& s, T val) {
			
		}
		if (boost::is_type<T, int>::value())
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