#include "Kernel.hpp"
#include "ArrayPool.hpp"
#include "utils/utils.hpp"
#include "Internal.hpp"

namespace oa {
	namespace kernel {
		// return u + v

		ArrayPtr kernel_plus(vector<ArrayPtr> &ops_ap) {
			ArrayPtr u = ops_ap[0];
			ArrayPtr v = ops_ap[1];
			ArrayPtr ap;
			
			int u_dt = u->get_data_type();
			int v_dt = v->get_data_type();
			int dt = oa::utils::cast_data_type(u_dt, v_dt);

			int case_num = dt * 100 + u_dt * 10 + v_dt;
			
#:mute
#:set i = 0
#:include "kernel_type.fypp"
#:endmute
	//create switch case
			switch(case_num) {
#:for i in L
#:set id = i[0]
#:set type1 = i[1]
#:set type2 = i[2]
#:set type3 = i[3]
				case ${id}$:
					ap = t_kernel_plus<${type1}$, ${type2}$, ${type3}$>(ops_ap);
					break;
#:endfor
			}			

			return ap;
		}

		
		

		// return u - v
		ArrayPtr kernel_minus(vector<ArrayPtr> &ops_ap) {
			
		}

		// return u * v
		ArrayPtr kernel_mult(vector<ArrayPtr> &ops_ap) {

		}

		// return u / v
		ArrayPtr kernel_divd(vector<ArrayPtr> &ops_ap) {

		}


	}
}
