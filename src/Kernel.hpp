/*
 * Kernel.hpp
 * kernel function declarations
 *
=======================================================*/

#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
#include "Internal.hpp"
#include <vector>
using namespace std;

#include "modules/mat_mult/kernel.hpp"
#include "modules/min_max/kernel.hpp"
#include "modules/sum/kernel.hpp"
#include "modules/interpolation/kernel.hpp"
#include "modules/set/kernel.hpp"
#include "modules/tree_tool/kernel.hpp"
#include "modules/sub/kernel.hpp"
#include "modules/basic/kernel.hpp"
#include "modules/shift/kernel.hpp"
#include "modules/rep/kernel.hpp"
#include "modules/operator/kernel.hpp"

namespace oa {
	namespace kernel {

		// return ANS = 
		ArrayPtr kernel_unknown(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_data(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_ref(vector<ArrayPtr> &ops_ap);

		// return ANS = sum
		ArrayPtr kernel_sum(vector<ArrayPtr> &ops_ap);

		// return ANS = csum(A)
		ArrayPtr kernel_csum(vector<ArrayPtr> &ops_ap);

		// return ANS = .not.B
		ArrayPtr kernel_not(vector<ArrayPtr> &ops_ap);

		// return ANS = rep(A)
		ArrayPtr kernel_rep(vector<ArrayPtr> &ops_ap);

		// return ANS = shift(A)
		ArrayPtr kernel_shift(vector<ArrayPtr> &ops_ap);

		// return ANS = circshift(A)
		ArrayPtr kernel_circshift(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_type_int(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_type_float(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_type_double(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_type_int3_rep(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_type_int3_rep(vector<ArrayPtr> &ops_ap);

		// return ANS = 
		ArrayPtr kernel_set(vector<ArrayPtr> &ops_ap);


	}
}
#endif

