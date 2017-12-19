#include "Partition.hpp"
#include "PartitionPool.hpp"
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <functional>
#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

// Partition has default parameters
Partition::Partition() { }
  
// give global_shape, and total process number
// find the approriate partition
Partition::Partition(MPI_Comm comm, int size, const Shape& gs, int sw) :
  m_comm(comm), m_global_shape(gs), m_stencil_width(sw) {
  assert(gs[0] > 0 && gs[1] > 0 && gs[2] > 0 && size > 0);
  double tot = pow(gs[0] * gs[1] * gs[2] * 1.0, 1.0 / 3);
  double fx = gs[0] / tot;
  double fy = gs[1] / tot;
  double fz = gs[2] / tot;

  //m_procs_shape = {-1, -1, -1};
  m_procs_shape = Partition::get_default_procs_shape();
  if(m_procs_shape[0] < 1){
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
            //cout<<factor<<" "<<new_factor<<" "<<i<<" "<<j<<" "<<k<<endl;
            if (factor >= new_factor) {
              m_procs_shape = {i, j, k};
              factor = new_factor;
            }
          }
      }
    assert(m_procs_shape[0] > 0);
  }
  
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
Partition::Partition(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
  const vector<int> &z, int sw) :
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
bool Partition::equal(const PartitionPtr &par_ptr) {
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
bool Partition::equal(const Partition &par) {
  PartitionPtr par_ptr = make_shared<Partition>(par);
  return equal(par_ptr);
}

// check if two Partition distribution is equal or not
bool Partition::equal_distr(const PartitionPtr &par_ptr) {
  if (m_lx.size() != par_ptr->m_lx.size() || m_ly.size() != par_ptr->m_ly.size() ||
  m_lz.size() != par_ptr->m_lz.size()) return false;

  for (int i = 0; i < m_lx.size(); i++) if (m_lx[i] != par_ptr->m_lx[i]) return false;
  for (int i = 0; i < m_ly.size(); i++) if (m_ly[i] != par_ptr->m_ly[i]) return false;
  for (int i = 0; i < m_lz.size(); i++) if (m_lz[i] != par_ptr->m_lz[i]) return false;
  return true;
}

// return global shape of Partition
Shape Partition::shape() {
  return m_global_shape;
}

Shape Partition::procs_shape() const{
  return m_procs_shape;
}

int Partition::procs_size() const {
  return m_procs_shape[0] * m_procs_shape[1] * m_procs_shape[2];
}

// return global size of Partition
int Partition::size() {
  return m_global_shape[0] * m_global_shape[1] * m_global_shape[2];
}

int Partition::rank() {
  int rank;
  MPI_Comm_rank(m_comm, &rank);
  return rank;
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

// get the box info
Box Partition::get_local_box() {
  int rk = rank();
  return get_local_box(rk);
}

// get the box info based on process's coord [px, py, pz]
Box Partition::get_local_box(const vector<int> &coord) {
  for (int i = 0; i < 3; i++) assert(0 <= coord[i] && coord[i] < m_procs_shape[i]);
  Box box(m_clx[coord[0]], m_clx[coord[0] + 1] - 1, 
  m_cly[coord[1]], m_cly[coord[1] + 1] - 1,
  m_clz[coord[2]], m_clz[coord[2] + 1] - 1);
  return box;
}

// get the box info based on process's rank
Box Partition::get_local_box(int rank) {
  vector<int> coord = get_procs_3d(rank);
  return get_local_box(coord);
}

// coord = [x, y, z], rank = x + y * px + z * px * py
int Partition::get_procs_rank(int x, int y, int z) {
  return x + y * m_procs_shape[0] + z * m_procs_shape[0] * m_procs_shape[1];
}

// coord = [x, y, z], rank = x + y * px + z * px * py
int Partition::get_procs_rank(const vector<int> &coord) {
  return get_procs_rank(coord[0], coord[1], coord[2]);
}

// given rank, calculate coord[x, y, z]
vector<int> Partition::get_procs_3d(int rank) {
  vector<int> coord(3, 0);
  coord[0] = rank % m_procs_shape[0];
  coord[1] = rank % (m_procs_shape[0] * m_procs_shape[1]) / m_procs_shape[0];
  coord[2] = rank / (m_procs_shape[0] * m_procs_shape[1]);
  return coord;
}

// display Partition information, default display all information
void Partition::display(const char *prefix, bool all) {
  if(prefix != NULL)
    printf("Partition %s:\n", prefix);    

  printf("\tglobal_shape = [%d, %d, %d]\n", 
  m_global_shape[0], m_global_shape[1], m_global_shape[2]);
  printf("\tprocs_shape = [%d, %d, %d]\n", 
  m_procs_shape[0], m_procs_shape[1], m_procs_shape[2]);
  printf("\tbound_type = [%d, %d, %d]\n", 
  m_bound_type[0], m_bound_type[1], m_bound_type[2]);
  printf("\tstencil_type = %d\n", m_stencil_type);
  printf("\tstencil_width = %d\n", m_stencil_width);

  if (all) display_distr(NULL);
}

// display distribution information
void Partition::display_distr(const char *prefix) {
  if(prefix != NULL)
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

void Partition::set_hash(const size_t &hash) {
  m_hash = hash;  
}

size_t Partition::get_hash() const {
  return m_hash;
}

MPI_Comm Partition::get_comm() const {
  return m_comm;
}

int Partition::get_stencil_width() const {
  return m_stencil_width;
}

void Partition::set_stencil_type(int st) {
  m_stencil_type = st;
}
int Partition::get_stencil_type() const {
  return m_stencil_type;
}

Shape Partition::get_bound_type() const {
  return m_bound_type;
}

// splic box in each procs
// rsx[procs_x * 3] box have procs_x processes in x dimension
// rsy[procs_y * 3]
// rsz[procs_z * 3]
void Partition::split_box_procs(const Box& b,
        vector<int> &rsx,
        vector<int> &rsy,
        vector<int> &rsz) const {
  
  int xs, ys, zs, xe, ye, ze;
  b.get_corners(xs, xe, ys, ye, zs, ze);
  
  int bxs = std::lower_bound(m_clx.begin(), m_clx.end(), xs) - m_clx.begin();
  int bxe = std::upper_bound(m_clx.begin(), m_clx.end(), xe - 1) - m_clx.begin();
  if (xs < m_clx[bxs]) bxs--;
  if (xe - 1 < m_clx[bxe]) bxe--;

  int bys = std::lower_bound(m_cly.begin(), m_cly.end(), ys) - m_cly.begin();
  int bye = std::upper_bound(m_cly.begin(), m_cly.end(), ye - 1) - m_cly.begin();
  if (ys < m_cly[bys]) bys--;
  if (ye - 1 < m_cly[bye]) bye--;

  int bzs = std::lower_bound(m_clz.begin(), m_clz.end(), zs) - m_clz.begin();
  int bze = std::upper_bound(m_clz.begin(), m_clz.end(), ze - 1) - m_clz.begin();
  if (zs < m_clz[bzs]) bzs--;
  if (ze - 1 < m_clz[bze]) bze--;

  //cout<<bxs<<" "<<bxe<<" "<<bys<<" "<<bye<<" "<<bzs<<" "<<bze<<endl;

  for(int i = bxs; i <= bxe; ++i) {
    rsx.push_back(std::max(xs, m_clx[i]));
    rsx.push_back(std::min(xe, m_clx[i+1]));
    rsx.push_back(i);
  }

  for(int i = bys; i <= bye; ++i) {
    rsy.push_back(std::max(ys, m_cly[i]));
    rsy.push_back(std::min(ye, m_cly[i+1]));
    rsy.push_back(i);
  }

  for(int i = bzs; i <= bze; ++i) {
    rsz.push_back(std::max(zs, m_clz[i]));
    rsz.push_back(std::min(ze, m_clz[i+1]));
    rsz.push_back(i);
  }
}

void Partition::get_acc_box_procs(vector<int> &rsx, vector<int> &rsy, vector<int> &rsz,
  vector<int> &acc_rsx, vector<int> &acc_rsy, vector<int> &acc_rsz) const {
  acc_rsx.push_back(0);
  acc_rsy.push_back(0);
  acc_rsz.push_back(0);

  for (int i = 0; i < rsx.size(); i += 3)
    acc_rsx.push_back(acc_rsx[i / 3] + rsx[i + 1] - rsx[i]);
  for (int i = 0; i < rsy.size(); i += 3)
    acc_rsy.push_back(acc_rsy[i / 3] + rsy[i + 1] - rsy[i]);
  for (int i = 0; i < rsz.size(); i += 3)
    acc_rsz.push_back(acc_rsz[i / 3] + rsz[i + 1] - rsz[i]);
}

PartitionPtr Partition::sub(const Box& b) const {
  vector<int> rsx, rsy, rsz;
  split_box_procs(b, rsx, rsy, rsz);
  //for (int i = 0; i < rsx.size(); i++) {
  //  cout<<rsx[i]<<" ";
  //  if (i % 3 == 2) cout<<endl;
  //}
  //cout<<endl;
  //for (int i = 0; i < rsy.size(); i++) {
  //  cout<<rsy[i]<<" ";
  //  if (i % 3 == 2) cout<<endl;
  //}
  //cout<<endl;
  //for (int i = 0; i < rsz.size(); i++) {
  //  cout<<rsz[i]<<" ";
  //  if (i % 3 == 2) cout<<endl;
  //}
  vector<int> x(m_procs_shape[0], 0), y(m_procs_shape[1], 0), z(m_procs_shape[2], 0);
  for (int i = 0; i < rsx.size(); i += 3)
    x[rsx[i + 2]] = rsx[i + 1] - rsx[i];
  for (int i = 0; i < rsy.size(); i += 3)
    y[rsy[i + 2]] = rsy[i + 1] - rsy[i];
  for (int i = 0; i < rsz.size(); i += 3)
    z[rsz[i + 2]] = rsz[i + 1] - rsz[i];
  
  PartitionPtr pp = PartitionPool::global()->
    get(m_comm, x, y, z, m_stencil_width);
  return pp;
}


// static function, gen hash based on [comm, gs(x, y, z), stencil_width]
size_t Partition::gen_hash(MPI_Comm comm, const Shape &gs, int stencil_width) {
  std::hash<string> str_hash;
  std::stringstream sstream;
  
  sstream<<comm<<":";
  sstream<<gs[0];
  for (int i = 1; i < gs.size(); i++) 
  sstream<<","<<gs[i];
  
  sstream<<":"<<stencil_width;
  //cout<<"gen_hash 1: "<<sstream.str()<<endl;
  return str_hash(sstream.str());  
}

// static function, gen hash based on [comm, x, y, z, stencil_width]
size_t Partition::gen_hash(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
  const vector<int> &z, int stencil_width) {
  std::hash<string> str_hash;
  std::stringstream sstream;

  sstream<<comm<<":";
  sstream<<x[0];
  for (int i = 1; i < x.size(); i++)
  sstream<<","<<x[i];

  sstream<<":"<<y[0];
  for (int i = 1; i < y.size(); i++) 
  sstream<<","<<y[i];

  sstream<<":"<<z[0];
  for (int i = 1; i < z.size(); i++)
  sstream<<","<<z[i];

  sstream<<":"<<stencil_width;

  return str_hash(sstream.str());
}

//initialize the default process shape to invilid shape
Shape Partition::m_default_procs_shape = {0,0,0};

Shape Partition::get_default_procs_shape(){
  return m_default_procs_shape;
}

void Partition::set_default_procs_shape(const Shape& s){
  m_default_procs_shape = s;
}

void Partition::set_auto_procs_shape(){
  m_default_procs_shape = {0,0,0};
}

int Partition::m_default_stencil_width = 1;
int Partition::get_default_stencil_width(){
  return m_default_stencil_width;
}

void Partition::set_default_stencil_width(int sw){
  m_default_stencil_width = sw;
}



