#include "Partition.hpp"
#include<assert.h>
#include<math.h>
#include<limits.h>
using namespace std;

/*
 * Partition has default parameters
 */
Partition :: Partition() { }
    
/*
 * give global_shape, and total process number
 * find the approriate partition
 */
Partition :: Partition(MPI_Comm comm, int size, vector<int> gs) :
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
                if (factor >= new_factor) {
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

/*
 * Give partition information, calculate procs_shape & global_shape
 */
Partition :: Partition(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z) :
    comm(comm), lx(x), ly(y), lz(z) {
        assert(lx.size() && ly.size() && lz.size());
        procs_shape[0] = lx.size();
        procs_shape[1] = ly.size();
        procs_shape[2] = lz.size();
        update_acc_distr();
        global_shape[0] = clx[procs_shape[0]];
        global_shape[1] = cly[procs_shape[1]];
        global_shape[2] = clz[procs_shape[2]];
}
    
// check if two Partition is equal or not
bool Partition :: equal(PartitionPtr par_ptr) {
    if (comm != par_ptr->comm || stencil_type != par_ptr->stencil_type || 
        stencil_width != par_ptr->stencil_width) return false;
    for (int i = 0; i < 3; i++) {
        if (global_shape[i] != par_ptr->global_shape[i] ||
            procs_shape[i] != par_ptr->procs_shape[i] ||
            bound_type[i] != par_ptr->bound_type[i]) return false;
    }
    return equal_distr(par_ptr);
}

bool Partition :: equal(Partition &par) {
    PartitionPtr par_ptr = make_shared<Partition>(par);
    return equal(par_ptr);
}

// check if two Partition distribution is equal or not
bool Partition :: equal_distr(PartitionPtr par_ptr) {
    if (lx.size() != par_ptr->lx.size() || ly.size() != par_ptr->ly.size() ||
        lz.size() != par_ptr->lz.size()) return false;

    for (int i = 0; i < lx.size(); i++) if (lx[i] != par_ptr->lx[i]) return false;
    for (int i = 0; i < ly.size(); i++) if (ly[i] != par_ptr->ly[i]) return false;
    for (int i = 0; i < lz.size(); i++) if (lz[i] != par_ptr->lz[i]) return false;
    return true;
}
// return global shape of Partition
vector<int> Partition :: shape() {
    return global_shape;
}

// return global size of Partition
int Partition :: size() {
    return global_shape[0] * global_shape[1] * global_shape[2];
}

void Partition :: update_acc_distr() {
    if (clx.size() != lx.size() + 1) clx = vector<int> (lx.size() + 1, 0);
    if (cly.size() != ly.size() + 1) cly = vector<int> (ly.size() + 1, 0);
    if (clz.size() != lz.size() + 1) clz = vector<int> (lz.size() + 1, 0);

    for (int i = 1; i < clx.size(); i++) clx[i] = clx[i - 1] + lx[i - 1];
    for (int i = 1; i < cly.size(); i++) cly[i] = cly[i - 1] + ly[i - 1];
    for (int i = 1; i < clz.size(); i++) clz[i] = clz[i - 1] + lz[i - 1];
}

// set stencil type & width
void Partition :: set_stencil(int type, int width) {
    stencil_type = type;
    stencil_width = width;
}

// get the box info based on process's coord [px, py, pz]
Box :: BoxPtr Partition :: get_local_box(vector<int> coord) {
    for (int i = 0; i < 3; i++) assert(0 <= coord[i] && coord[i] < procs_shape[i]);
    Box box(clx[coord[0]], clx[coord[0] + 1] - 1, 
            cly[coord[1]], cly[coord[1] + 1] - 1,
            clz[coord[2]], clz[coord[2] + 1] - 1);
    Box :: BoxPtr box_ptr = make_shared<Box>(box);
    return box_ptr;
}

// get the box info based on process's rank
Box :: BoxPtr Partition :: get_local_box(int rank) {
    vector<int> coord = get_procs_3d(rank);
    return get_local_box(coord);
}

// coord = [x, y, z], rank = x + y * px + z * px * py
int Partition :: get_procs_rank(int x, int y, int z) {
    return x + y * procs_shape[0] + z * procs_shape[0] * procs_shape[1];
}

int Partition :: get_procs_rank(vector<int> coord) {
    return get_procs_rank(coord[0], coord[1], coord[2]);
}

// given rank, calculate coord[x, y, z]
vector<int> Partition :: get_procs_3d(int rank) {
    vector<int> coord(3, 0);
    coord[0] = rank % global_shape[0];
    coord[1] = rank % (global_shape[0] * global_shape[1]) / global_shape[0];
    coord[2] = rank / (global_shape[0] * global_shape[1]);
    return coord;
}

void Partition :: display(const char *prefix, bool all) {
    printf("Partition %s:\n", prefix);
    printf("\tglobal_shape = [%d, %d, %d]\n", 
        global_shape[0], global_shape[1], global_shape[2]);
    printf("\tprocs_shape = [%d, %d, %d]\n", 
        procs_shape[0], procs_shape[1], procs_shape[2]);
    printf("\tbound_type = [%d, %d, %d]\n", 
        bound_type[0], bound_type[1], bound_type[2]);
    printf("\tstencil_type = %d\n", stencil_type);
    printf("\tstencil_width = %d\n", stencil_width);

    if (all) display_distr(prefix);
}

void Partition :: display_distr(const char *prefix) {
    printf("%s distr info\n", prefix);
    printf("\tlx = [%d", lx[0]);
    for (int i = 1; i < lx.size(); i++) printf(", %d", lx[i]);
    printf("]\n");
    printf("\tly = [%d", ly[0]);
    for (int i = 1; i < ly.size(); i++) printf(", %d", ly[i]);
    printf("]\n");
    printf("\tlz = [%d", lz[0]);
    for (int i = 1; i < lz.size(); i++) printf(", %d", lz[i]);
    printf("]\n");

    printf("\tclx = [%d", clx[0]);
    for (int i = 1; i < clx.size(); i++) printf(", %d", clx[i]);
    printf("]\n");
    printf("\tcly = [%d", cly[0]);
    for (int i = 1; i < cly.size(); i++) printf(", %d", cly[i]);
    printf("]\n");
    printf("\tclz = [%d", clz[0]);
    for (int i = 1; i < clz.size(); i++) printf(", %d", clz[i]);
    printf("]\n");        
}
