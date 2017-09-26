#include<mpi.h>
#include<vector>
#include<memory>
#include "type.hpp"
#include "Box.hpp"

using namespace std;

#ifndef PARTITION_HPP
#define PARTITION_HPP

/*
 * Partition:
 *      shape:          array global shape [gx, gy, gz]
 *      procs_shape:    processes shape [px, py, pz]
 *      lx, ly, lz:     partition info of every dimension
 *      clx, cly, clz:  accumulate info
 */

class Partition {
    private:
        MPI_Comm comm = MPI_COMM_SELF;
        vector<int> global_shape = {1, 1, 1};
        vector<int> procs_shape = {1, 1, 1};
        vector<int> bound_type = {0, 0, 0};
        int stencil_type = STENCIL_STAR;
        int stencil_width = 0;

        vector<int> lx = {1};
        vector<int> ly = {1};
        vector<int> lz = {1};
        vector<int> clx = {0, 1};
        vector<int> cly = {0, 1};
        vector<int> clz = {0, 1};
        
    public:
        typedef shared_ptr<Partition> PartitionPtr;
        Partition();
        Partition(MPI_Comm comm, int size, vector<int> gs);
        Partition(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z);
        bool equal(PartitionPtr par_ptr);
        bool equal(Partition &par);
        bool equal_distr(PartitionPtr par_ptr);
        vector<int> shape();
        int size();
        void update_acc_distr();
        void set_stencil(int type, int width);
        Box :: BoxPtr get_local_box(vector<int> coord);
        Box :: BoxPtr get_local_box(int rank);
        int get_procs_rank(int x, int y, int z);
        int get_procs_rank(vector<int> coord);
        vector<int> get_procs_3d(int rank);
        void display(const char *prefix = "", bool all = true); 
        void display_distr(const char *prefix = "");
};


#endif
