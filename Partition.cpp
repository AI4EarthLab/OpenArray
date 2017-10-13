#include "Partition.hpp"
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <functional>
#include <iostream>
using namespace std;

// Partition has default parameters
Partition::Partition() { }
    
// give global_shape, and total process number
// find the approriate partition
Partition::Partition(MPI_Comm comm, int size, vector<int> gs, int sw) :
    m_comm(comm), m_global_shape(gs), m_stencil_width(sw) {
        assert(gs[0] > 0 && gs[1] > 0 && gs[2] > 0 && size > 0);
        double tot = pow(gs[0] * gs[1] * gs[2] * 1.0, 1.0 / 3);
        double fx = gs[0] / tot;
        double fy = gs[1] / tot;
        double fz = gs[2] / tot;

        m_procs_shape = {-1, -1, -1};
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
                    m_procs_shape = {i, j, k};
                    factor = new_factor;
                }
            }
        }
        assert(m_procs_shape[0] > 0);
        
        m_lx = vector<int> (m_procs_shape[0], gs[0] / m_procs_shape[0]);
        m_ly = vector<int> (m_procs_shape[1], gs[1] / m_procs_shape[1]);
        m_lz = vector<int> (m_procs_shape[2], gs[2] / m_procs_shape[2]);
        
        int ed = gs[0] % m_procs_shape[0];
        for (int i = 0; i < ed; i++) m_lx[i] += 1;
        ed = gs[1] % m_procs_shape[1];
        for (int i = 0; i < ed; i++) m_ly[i] += 1;
        ed = gs[2] % m_procs_shape[2];
        for (int i = 0; i < ed; i++) m_lz[i] += 1;

        update_acc_distr();
}


// Give partition information, calculate procs_shape & global_shape
Partition::Partition(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z, int sw) :
    m_comm(comm), m_stencil_width(sw), m_lx(x), m_ly(y), m_lz(z) {
        assert(m_lx.size() && m_ly.size() && m_lz.size());
        m_procs_shape[0] = m_lx.size();
        m_procs_shape[1] = m_ly.size();
        m_procs_shape[2] = m_lz.size();
        update_acc_distr();
        m_global_shape[0] = m_clx[m_procs_shape[0]];
        m_global_shape[1] = m_cly[m_procs_shape[1]];
        m_global_shape[2] = m_clz[m_procs_shape[2]];
}
    
// check if two Partition is equal or not
bool Partition::equal(PartitionPtr par_ptr) {
    if (m_comm != par_ptr->m_comm || m_stencil_type != par_ptr->m_stencil_type || 
        m_stencil_width != par_ptr->m_stencil_width) return false;
    for (int i = 0; i < 3; i++) {
        if (m_global_shape[i] != par_ptr->m_global_shape[i] ||
            m_procs_shape[i] != par_ptr->m_procs_shape[i] ||
            m_bound_type[i] != par_ptr->m_bound_type[i]) return false;
    }
    return equal_distr(par_ptr);
}

// check if two Partition is equal or not
bool Partition::equal(Partition &par) {
    PartitionPtr par_ptr = make_shared<Partition>(par);
    return equal(par_ptr);
}

// check if two Partition distribution is equal or not
bool Partition::equal_distr(PartitionPtr par_ptr) {
    if (m_lx.size() != par_ptr->m_lx.size() || m_ly.size() != par_ptr->m_ly.size() ||
        m_lz.size() != par_ptr->m_lz.size()) return false;

    for (int i = 0; i < m_lx.size(); i++) if (m_lx[i] != par_ptr->m_lx[i]) return false;
    for (int i = 0; i < m_ly.size(); i++) if (m_ly[i] != par_ptr->m_ly[i]) return false;
    for (int i = 0; i < m_lz.size(); i++) if (m_lz[i] != par_ptr->m_lz[i]) return false;
    return true;
}

// return global shape of Partition
vector<int> Partition::shape() {
    return m_global_shape;
}

// return global size of Partition
int Partition::size() {
    return m_global_shape[0] * m_global_shape[1] * m_global_shape[2];
}

// update accumulate distribution
void Partition::update_acc_distr() {
    if (m_clx.size() != m_lx.size() + 1) m_clx = vector<int> (m_lx.size() + 1, 0);
    if (m_cly.size() != m_ly.size() + 1) m_cly = vector<int> (m_ly.size() + 1, 0);
    if (m_clz.size() != m_lz.size() + 1) m_clz = vector<int> (m_lz.size() + 1, 0);

    for (int i = 1; i < m_clx.size(); i++) m_clx[i] = m_clx[i - 1] + m_lx[i - 1];
    for (int i = 1; i < m_cly.size(); i++) m_cly[i] = m_cly[i - 1] + m_ly[i - 1];
    for (int i = 1; i < m_clz.size(); i++) m_clz[i] = m_clz[i - 1] + m_lz[i - 1];
}

// set stencil type & width
void Partition::set_stencil(int type, int width) {
    m_stencil_type = type;
    m_stencil_width = width;
}

// get the box info based on process's coord [px, py, pz]
BoxPtr Partition::get_local_box(vector<int> coord) {
    for (int i = 0; i < 3; i++) assert(0 <= coord[i] && coord[i] < m_procs_shape[i]);
    Box box(m_clx[coord[0]], m_clx[coord[0] + 1] - 1, 
            m_cly[coord[1]], m_cly[coord[1] + 1] - 1,
            m_clz[coord[2]], m_clz[coord[2] + 1] - 1);
    BoxPtr box_ptr = make_shared<Box>(box);
    return box_ptr;
}

// get the box info based on process's rank
BoxPtr Partition::get_local_box(int rank) {
    vector<int> coord = get_procs_3d(rank);
    return get_local_box(coord);
}

// coord = [x, y, z], rank = x + y * px + z * px * py
int Partition::get_procs_rank(int x, int y, int z) {
    return x + y * m_procs_shape[0] + z * m_procs_shape[0] * m_procs_shape[1];
}

// coord = [x, y, z], rank = x + y * px + z * px * py
int Partition::get_procs_rank(vector<int> coord) {
    return get_procs_rank(coord[0], coord[1], coord[2]);
}

// given rank, calculate coord[x, y, z]
vector<int> Partition::get_procs_3d(int rank) {
    vector<int> coord(3, 0);
    coord[0] = rank % m_global_shape[0];
    coord[1] = rank % (m_global_shape[0] * m_global_shape[1]) / m_global_shape[0];
    coord[2] = rank / (m_global_shape[0] * m_global_shape[1]);
    return coord;
}

// display Partition information, default display all information
void Partition::display(const char *prefix, bool all) {
    printf("Partition %s:\n", prefix);
    printf("\tglobal_shape = [%d, %d, %d]\n", 
        m_global_shape[0], m_global_shape[1], m_global_shape[2]);
    printf("\tprocs_shape = [%d, %d, %d]\n", 
        m_procs_shape[0], m_procs_shape[1], m_procs_shape[2]);
    printf("\tbound_type = [%d, %d, %d]\n", 
        m_bound_type[0], m_bound_type[1], m_bound_type[2]);
    printf("\tstencil_type = %d\n", m_stencil_type);
    printf("\tstencil_width = %d\n", m_stencil_width);

    if (all) display_distr(prefix);
}

// display distribution information
void Partition::display_distr(const char *prefix) {
    printf("%s distr info\n", prefix);
    printf("\tlx = [%d", m_lx[0]);
    for (int i = 1; i < m_lx.size(); i++) printf(", %d", m_lx[i]);
    printf("]\n");
    printf("\tly = [%d", m_ly[0]);
    for (int i = 1; i < m_ly.size(); i++) printf(", %d", m_ly[i]);
    printf("]\n");
    printf("\tlz = [%d", m_lz[0]);
    for (int i = 1; i < m_lz.size(); i++) printf(", %d", m_lz[i]);
    printf("]\n");

    printf("\tclx = [%d", m_clx[0]);
    for (int i = 1; i < m_clx.size(); i++) printf(", %d", m_clx[i]);
    printf("]\n");
    printf("\tcly = [%d", m_cly[0]);
    for (int i = 1; i < m_cly.size(); i++) printf(", %d", m_cly[i]);
    printf("]\n");
    printf("\tclz = [%d", m_clz[0]);
    for (int i = 1; i < m_clz.size(); i++) printf(", %d", m_clz[i]);
    printf("]\n");
    //flush();        
}

// set partition hash
void Partition::set_hash(size_t hash) {
    m_hash = hash;    
}

// get partition hash
size_t Partition::hash() {
    return m_hash;
}

// static function, gen hash based on [comm, size, gs, stencil_width]
size_t Partition::gen_hash(MPI_Comm comm, int size, vector<int> gs, int stencil_width) {
    std::hash<string> str_hash;
    string str = "";
    str += to_string(comm) + ":";
    str += to_string(size) + ":";
    str += to_string(gs[0]);
    for (int i = 1; i < gs.size(); i++) str += "," + to_string(gs[i]);
    str += ":" + to_string(stencil_width);
    cout<<"gen_hash 1: "<<str<<endl;
    return str_hash(str);    
}

// static function, gen hash based on [comm, x, y, z, stencil_width]
size_t Partition::gen_hash(MPI_Comm comm, vector<int> x, vector<int> y, vector<int> z, int stencil_width) {
    std::hash<string> str_hash;
    string str = "";
    str += to_string(comm) + ":";
    str += to_string(x[0]);
    for (int i = 1; i < x.size(); i++) str += "," + to_string(x[i]);
    str += ":" + to_string(y[0]);
    for (int i = 1; i < y.size(); i++) str += "," + to_string(y[i]);
    str += ":" + to_string(z[0]);
    for (int i = 1; i < z.size(); i++) str += "," + to_string(z[i]);
    str += ":" + to_string(stencil_width);
    return str_hash(str);
}



