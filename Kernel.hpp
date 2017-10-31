#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
#include "Internal.hpp"
#include <vector>
using namespace std;

namespace oa {
	namespace kernel {
		// return u + v
		ArrayPtr kernel_plus(vector<ArrayPtr> &ops_ap);

		// return u - v
		ArrayPtr kernel_minus(vector<ArrayPtr> &ops_ap); 

		// return u * v
		ArrayPtr kernel_mult(vector<ArrayPtr> &ops_ap); 

		// return u / v
		ArrayPtr kernel_divd(vector<ArrayPtr> &ops_ap); 

		// ap = u {+ - * /} v
#:mute
#:include "NodeType.fypp"
#:endmute
#:for k in L[2:6]
#:set name = k[1]
#:set sy = k[2]
		// A = U ${sy}$ V
		template <typename T1, typename T2, typename T3>
		ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
			ArrayPtr u = ops_ap[0];
			ArrayPtr v = ops_ap[1];
			ArrayPtr ap;
			
			int u_dt = u->get_data_type();
			int v_dt = v->get_data_type();
			int dt = oa::utils::cast_data_type(u_dt, v_dt);

			if (u->is_seqs_scalar()) {
				ap = ArrayPool::global()->get(v->get_partition(), dt);
				T2 scalar = *(T2*) u->get_buffer();
				oa::internal::buffer_${name}$_const(
					(T1 *) ap->get_buffer(),
					(T3 *) v->get_buffer(),
					scalar,
					ap->buffer_size()
				);
			} else if (v->is_seqs_scalar()) {
				ap = ArrayPool::global()->get(u->get_partition(), dt);
				T3 scalar = *(T3*) v->get_buffer();
				oa::internal::buffer_${name}$_const(
					(T1 *) ap->get_buffer(),
					(T2 *) u->get_buffer(),
					scalar,
					ap->buffer_size()
				);
			} else {
				PartitionPtr upar = u->get_partition();
				PartitionPtr vpar = v->get_partition();
				assert(upar->get_comm() == vpar->get_comm());

/*				// U and V must have same shape
				assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
*/
				ap = ArrayPool::global()->get(upar, dt);
				if (upar->equal(vpar)) {
					oa::internal::buffer_${name}$_buffer(
						(T1 *) ap->get_buffer(),
						(T2 *) u->get_buffer(),
						(T3 *) v->get_buffer(),
						ap->buffer_size()
					);
				} else {
					ArrayPtr tmp = oa::funcs::transfer(v, upar);
					oa::internal::buffer_${name}$_buffer(
						(T1 *) ap->get_buffer(),
						(T2 *) u->get_buffer(),
						(T3 *) tmp->get_buffer(),
						ap->buffer_size()
					);
				}
			}
			return ap;
		}

#:endfor
	}
}

#endif
