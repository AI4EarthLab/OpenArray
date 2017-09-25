#include "Partition.hpp"
#include<assert.h>
#include<math.h>
#include<limits.h>
using namespace std;

Partition :: Partition() { 
    update_acc_distr();
}
    
/*
 * give global_shape, and total process number
 * find the approriate partition
 */
Partition :: Partition(MPI_Comm &comm, int size, vector<int> &gs) :
    comm(comm), global_shape(gs) {
        assert(gs[0] > 0 && gs[1] > 0 && gs[2] > 0 && size > 0);
        double tot = pow(gs[0] * gs[1] * gs[2] * 1.0, 1.0 / 3);
        double fx = gs[0] / tot;
        double fy = gs[1] / tot;
        double fz = gs[2] / tot;

        procs_shape = {-1, -1, -1};
        double factor = INT_MAX;
        double tsz = pow(size * 1.0, 1.0 / 3);

        for (int i = 1; i <= size; i++) if (size % i == 0) {
            int ed = size / i;
            for (int j = 1; j <= ed; j++) if (ed % j == 0) {
                int k = ed / j;
                double dfx = fx - i * 1.0 / tsz;
                double dfy = fy - j * 1.0 / tsz;
                double dfz = fz - k * 1.0 / tsz;
                double new_factor = dfx * dfx + dfy * dfy + dfz * dfz;
                if (factor < new_factor) {
                    procs_shape = {i, j, k};
                    factor = new_factor;
                }
            }
        }
        
        assert(procs_shape[0] > 0);
        
        lx = vector<int> (procs_shape[0], gs[0] / procs_shape[0]);
        ly = vector<int> (procs_shape[1], gs[1] / procs_shape[1]);
        lz = vector<int> (procs_shape[2], gs[2] / procs_shape[2]);
        
        int ed = gs[0] % procs_shape[0];
        for (int i = 0; i < ed; i++) lx[i] += 1;
        ed = gs[1] % procs_shape[1];
        for (int i = 0; i < ed; i++) ly[i] += 1;
        ed = gs[2] % procs_shape[2];
        for (int i = 0; i < ed; i++) lz[i] += 1;

        update_acc_distr();
}
//        Partition(MPI_Comm &comm, vector<int> &x, vector<int> &y, vector<int> &z);
//        
//        // check if two Partition is equal or not
//        bool equal(PartitionPtr par_ptr);
//        bool equal_distr(PartitionPtr par_ptr);
//        
//        // return global shape of Partition
//        vector<int> shape();
//
//        // return global size of Partition
//        int size();
//        void update_distr();
//void Partition :: update_acc_distr() {
//    
//}
//        void set_stencil(int type, int width);
//        Box :: BoxPtr get_local_box(vector<int> &coord);
//        Box :: BoxPtr get_local_box(int rank);
//        int get_procs_rank(vector<int> &coord);
//        int get_procs_rank(int x, int y, int z);
//        vector<int> get_procs_3d(int rank);

