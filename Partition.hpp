#ifndef __PARTITION_HPP__
#define __PARTITION_HPP__

#include <mpi.h>
#include <vector>
#include <memory>
#include <list>
#include "common.hpp"
#include "Box.hpp"

using namespace std;

/*
 * Partition:
 *  comm:   MPI_Comm, default as MPI_COMM_SELF
 *  global_shape:   array global shape [gx, gy, gz], default as [1, 1, 1]
 *  procs_shape:  processes shape [px, py, pz], default as [1, 1, 1]
 *  bound_type:   bound_type [bx, by, bz], default as [0, 0, 0]
 *  stencil_type:   array stencil type, default as STENCIL_STAR
 *  stencil_width:  array stencil width, default as 1
 *  hash:   each partiton has it's hash, used in PartitionPool
 *  lx, ly, lz:   partition info of every dimension
 *  clx, cly, clz:  accumulate info of every dimension
 */
class Partition;
typedef std::shared_ptr<Partition> PartitionPtr;

class Partition {
private:
  static Shape m_default_procs_shape;
  static int m_default_stencil_width;
  static int m_default_stencil_type;
  public:
  MPI_Comm m_comm = MPI_COMM_SELF;
  Shape m_global_shape = { {1, 1, 1} };
  Shape m_procs_shape = {{1, 1, 1}};
  Shape m_bound_type = {{1, 1, 1}};
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
  Partition(MPI_Comm comm, int size,
          const Shape &gs, int stencil_width = 1);
  
  Partition(MPI_Comm comm, const vector<int> &x,
          const vector<int> &y, 
    const vector<int> &z, int stencil_width = 1);
  bool equal(const PartitionPtr &par_ptr);
  bool equal(const Partition &par);
  bool equal_distr(const PartitionPtr &par_ptr);
  Shape shape();
  int size();
  int rank();
  Shape procs_shape() const;
  int procs_size() const;
  void update_acc_distr();
  void set_stencil(int type, int width);
  Box get_local_box();
  Box get_local_box(const vector<int> &coord);
  Box get_local_box(int rank);
  int get_procs_rank(int x, int y, int z);
  int get_procs_rank(const vector<int> &coord);
  vector<int> get_procs_3d(int rank);
  void display(const char *prefix = NULL, bool all = true); 
  void display_distr(const char *prefix = NULL);
  void set_hash(const size_t &hash);
  size_t get_hash() const;
  MPI_Comm get_comm() const;
  int get_stencil_width() const;
  void set_stencil_type(int st);
  int get_stencil_type() const;
  Shape get_bound_type() const;

  void split_box_procs(const Box& b,
        vector<int> &rsx,
        vector<int> &rsy,
        vector<int> &rsz) const;

  void get_acc_box_procs(
    vector<int> &rsx, 
    vector<int> &rsy, 
    vector<int> &rsz,
    vector<int> &acc_rsx, 
    vector<int> &acc_rsy, 
    vector<int> &acc_rsz) const;
  
  PartitionPtr sub(const Box& b) const;

  static size_t gen_hash(MPI_Comm comm, const Shape& gs, int stencil_width = 1);
  static size_t gen_hash(MPI_Comm comm, const vector<int> &x, 
    const vector<int> &y, const vector<int> &z, int stencil_width = 1);

  static void set_default_procs_shape(const Shape& s);
  static void set_auto_procs_shape();  
  static Shape get_default_procs_shape();

  static int  get_default_stencil_width();
  static void set_default_stencil_width(int);

  static int  get_default_stencil_type();
  static void set_default_stencil_type(int);  
};


#endif
