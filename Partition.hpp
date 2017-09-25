#include<mpi.h>
#include<vector>
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
        MPI_Comm comm;
        vector<int> shape;
        vector<int> procs_shape;
        vector<BoundType> bound_type;
        StencilType stencil_type;
        int stencil_width;
        
        vector<int> lx;
        vector<int> ly;
        vector<int> lz;
        vector<int> clx;
        vector<int> cly;
        vector<int> clz;
        

};

#endif
