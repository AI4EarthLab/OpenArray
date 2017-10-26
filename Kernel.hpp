#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
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
		template <typename T1, typename T2, typename T3>
		ArrayPtr t_kernel_plus(vector<ArrayPtr> &ops_ap) {
			ArrayPtr u = ops_ap[0];
			ArrayPtr v = ops_ap[1];
			ArrayPtr ap;
			
			int u_dt = u->get_data_type();
			int v_dt = v->get_data_type();
			int dt = oa::utils::cast_data_type(u_dt, v_dt);

			if (u->is_seqs_scalar()) {
				ap = ArrayPool::global()->get(v->get_partition(), dt);
				oa::internal::buffer_plus_const(
					(T1 *) ap->get_buffer(),
					(T3 *) v->get_buffer(),
					*(T2 *) u->get_scalar(),
					ap->buffer_size()
				);
			} else if (v->is_seqs_scalar()) {
				ap = ArrayPool::global()->get(u->get_partition(), dt);
				oa::internal::buffer_plus_const(
					(T1 *) ap->get_buffer(),
					(T2 *) u->get_buffer(),
					*(T3 *) v->get_scalar(),
					ap->buffer_size()
				);
			} else {
				PartitionPtr upar = u->get_partition();
				PartitionPtr vpar = v->get_partition();
				assert(upar->get_comm() == vpar->get_comm());

				ap = ArrayPool::global()->get(upar, dt);
				if (upar->equal(vpar)) {
					oa::internal::buffer_plus_buffer(
						(T1 *) ap->get_buffer(),
						(T2 *) u->get_buffer(),
						(T3 *) v->get_buffer(),
						ap->buffer_size()
					);
				} else {
					ArrayPtr tmp = oa::funcs::transfer(v, upar);
					oa::internal::buffer_plus_buffer(
						(T1 *) ap->get_buffer(),
						(T2 *) u->get_buffer(),
						(T3 *) tmp->get_buffer(),
						ap->buffer_size()
					);
				}
			}
			return ap;
		}

	}
}

#endif