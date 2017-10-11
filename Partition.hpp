#ifndef PARTITION_HPP
#define PARTITION_HPP

#include <mpi.h>
#include <vector>
#include <memory>
#include <list>
#include "type.hpp"
#include "Box.hpp"

/*
 * Partition:
 *      shape:          array global shape [gx, gy, gz]
 *      procs_shape:    processes shape [px, py, pz]
 *      lx, ly, lz:     partition info of every dimension
 *      clx, cly, clz:  accumulate info
 */
class Partition;
typedef shared_ptr<Partition> PartitionPtr;
typedef list<Partition> PartitionList;

class Partition {
    private:
        MPI_Comm m_comm = MPI_COMM_SELF;
        vector<int> m_global_shape = {1, 1, 1};
        vector<int> m_procs_shape = {1, 1, 1};
        vector<int> m_bound_type = {0, 0, 0};
        int m_stencil_type = STENCIL_STAR;
        int m_stencil_width = 1;
        size_t m_hash;

        vector<int> m_lx = {1};
        vector<int> m_ly = {1};
        vector<int> m_lz = {1};
        vector<int> m_clx = {0, 1};
        vector<int> m_cly = {0, 1};
        vector<int> m_clz = {0, 1};
    
    public:
        static const PartitionPtr Scalar;
    
    public:
        Partition();
        Partition(MPI_Comm comm, int size, vector<int> gs, int stencil_width = 1);
        Partition(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z, int stencil_width = 1);
        bool equal(PartitionPtr par_ptr);
        bool equal(Partition &par);
        bool equal_distr(PartitionPtr par_ptr);
        vector<int> shape();
        int size();
        void update_acc_distr();
        void set_stencil(int type, int width);
        BoxPtr get_local_box(vector<int> coord);
        BoxPtr get_local_box(int rank);
        int get_procs_rank(int x, int y, int z);
        int get_procs_rank(vector<int> coord);
        vector<int> get_procs_3d(int rank);
        void display(const char *prefix = "", bool all = true); 
        void display_distr(const char *prefix = "");

        static size_t gen_hash(MPI_Comm comm, int size, vector<int> gs, int stencil_width = 1);
        static size_t gen_hash(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z, int stencil_width = 1);
};


#endif
