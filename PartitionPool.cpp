
#ifndef __PARTITIONPOOL_HPP__
#define __PARTITIONPOOL_HPP__

#include "Partition.hpp"
#include <unordered_map>
#include <vector>

using namespace std;

typedef unordered_map<size_t, PartitionPtr> PartitonPoolMap;

/*
 * Partition Pool:
 *      m_pool:    		use [comm, process_size, (gx, gy, gz), stencil_width] as identicial key
 *		m_pool_xyz:		use [comm, lx, ly, lz, stencil_width] as identical key
 */
class PartitionPool {
	private:
		PartitionPoolMap m_pool;
		PartitionPoolMap m_pool_xyz;

	public:
		PartitionPtr get(MPI_Comm comm, int size, vector<int> gs, int stencil_width = 1) {
			PartitionPtr par_ptr;
			size_t par_hash = Partition :: gen_hash(comm, size, gs, stencil_width);

			PartitionPoolMap :: iterator it = m_pool.find(par_hash);
			if (it == m_pool.end()) {
				par_ptr = PartitionPtr(new Partition(comm, size, gs, stencil_width));
				m_pool[size_t] = par_ptr;
			} else {
				par_ptr = *it;
			}
			return par_ptr;
		}

		PartitionPtr get(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z, int stencil_width = 1) {
			PartitionPtr par_ptr;
			size_t par_hash = Partition :: gen_hash(comm, x, y, z, stencil_width);

			PartitionPoolMap :: iterator it = m_pool_xyz.find(par_hash);
			if (it == m_pool_xyz.end()) {
				par_ptr = PartitionPtr(new Partition(comm, x, y, z, stencil_width));
				m_pool_xyz[size_t] = par_ptr;
			} else {
				par_ptr = *it;
			}
			return par_ptr;
		}
		

};
#endif
