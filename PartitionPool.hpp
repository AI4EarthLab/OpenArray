
#ifndef __PARTITIONPOOL_HPP__
#define __PARTITIONPOOL_HPP__

#include "Partition.hpp"
#include <unordered_map>
#include <vector>

using namespace std;

typedef unordered_map<size_t, PartitionPtr> PartitionPoolMap;

/*
 * Partition Pool:
 *  m_pool:     use [comm, procs_size, (gx, gy, gz), stencil_width] as identicial key
 *    m_pool_xyz:   use [comm, lx, ly, lz, stencil_width] as identical key
 */
class PartitionPool {
  private:
    PartitionPoolMap m_pool;
    PartitionPoolMap m_pool_xyz;
    int global_count = 0;

  public:
    // get a PartitionPtr from m_pool based on hash created by key:
  // [comm, process_size, (gx, gy, gz), stencil_width] 
  PartitionPtr get(MPI_Comm comm, int size, const Shape& gs,
          int stencil_width = 1, size_t par_hash = 0) {
    PartitionPtr par_ptr;
      // par_hash == 0: should gen hash
      // par_hash != 0: means arraypool has already gen hash, just use it
      
      if (par_hash == 0)
        par_hash = Partition::gen_hash(comm, gs, stencil_width);
      
      PartitionPoolMap::iterator it = m_pool.find(par_hash);
      if (it == m_pool.end()) { // create new partition in pool
        par_ptr = PartitionPtr(new Partition(comm, size, gs, stencil_width));
        add_count();
        if (g_debug) cout<<"PartitionPool.size() = "<<count()<<endl;
        par_ptr->set_hash(par_hash);
        m_pool[par_hash] = par_ptr;
      } else { // get partition from pool
        par_ptr = it->second;
      }
      return par_ptr;
    }

    // get a PartitionPtr from m_pool_xyz based on hash created by key:
    // [comm, lx, ly, lz, stencil_width]
    PartitionPtr get(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
      const vector<int> &z, int stencil_width = 1, size_t par_hash = 0) {
      PartitionPtr par_ptr;
      // par_hash == 0: should gen hash
      // par_hash != 0: means arraypool has already gen hash, just use it
      if (par_hash == 0) par_hash = Partition::gen_hash(comm, x, y, z, stencil_width);

      PartitionPoolMap::iterator it = m_pool_xyz.find(par_hash);
      if (it == m_pool_xyz.end()) { // create new partition in pool
        par_ptr = PartitionPtr(new Partition(comm, x, y, z, stencil_width));
        add_count();
        if (g_debug) cout<<"PartitionPool.size() = "<<count()<<endl;
        par_ptr->set_hash(par_hash);
        m_pool_xyz[par_hash] = par_ptr;
      } else { // get partition from pool
        par_ptr = it->second;
      }
      return par_ptr;
    }

    // only need one Partition Pool in each process, so make it static
    // static variable in .hpp
    static PartitionPool* global() {
      static PartitionPool par_pool;
      return &par_pool;
    }

    int count() {
      return global_count;
    }

    void add_count() {
      global_count += 1;
    }
};
#endif
