/*
 * Array.hpp
 * 
 *
=======================================================*/

#ifndef __ARRAY_HPP__
#define __ARRAY_HPP__

#include <vector>
#include <memory>
#include <bitset>
#include <string>
#include "Internal.hpp"
#include "Partition.hpp"

class Grid;
class Array;
typedef shared_ptr<Grid> GridPtr;
typedef shared_ptr<Array> ArrayPtr;

class Array {
  private:
  bool m_is_seqs = false;       // is sequence or not 
  bool m_is_scalar = false;     // is scalar or not
  bool m_is_pseudo = false;     // is pseudo array or not
  bool m_has_pseudo_3d = false; // has pseudo_3d array or not

  // flag for halo region, is updated or not,  not used yet
  bool m_lb_ghost_updated[3] = {false, false, false};
  bool m_rb_ghost_updated[3] = {false, false, false};

  int pos = -1;         // default grid pos is -1
  int m_data_type = 2;  // default data type is double(2)
  
  size_t m_hash;        // hash value for array, used in Array Pool
  void *m_buffer;       // array's data

  Box m_corners;            // array's local box
  BoxPtr m_ref_box_ptr;     // array's reference box
  ArrayPtr m_pseudo_3d;     // array's pseudo_3d
  PartitionPtr m_par_ptr;   // array's partition pointer

  // array's bitset, 110 means the size of z dimension is 1, default bitset is 111
  std::bitset<3> m_bs = std::bitset<3>(7);

  public:
  // Constructor, default data type is DATA_DOUBLE
  Array(const PartitionPtr &ptr, int data_type = DATA_DOUBLE); 

  // Destructor
  ~Array();

  // get array's data type
  int get_data_type() const;

  // return array data buffer's head pointer
  void* get_buffer();

  // get array's partition shared pointer
  PartitionPtr get_partition() const;

  // display array information without data
  void display_info(const char *prefix = "");

  // display array
  void display(const char *prefix = "", int is = -1, int ie = -1, int js = -1, int je = -1, int ks = -1, int ke = -1);

  // display for gdb
  void display_for_gdb();

  // set & get local box in each process
  void set_local_box();
  Box get_local_box() const;

  // return array's buffer shape including the stencil (per each process)
  Shape buffer_shape() const;

  // return array's buffer size including the stencil (per each process)
  int buffer_size() const;

  // return array's local shape not including the stencil (per each process)
  Shape local_shape();

  // return array's local size not including the stencil (per each process)
  int local_size() const;

  // return local box in the window of buffer shape
  Box local_data_win() const;

  Shape shape();    // return global shape of Array
  int size();       // return global size of Array
  int rank();       // return the process rank of Array

  bool is_scalar();       // Array is a scalar or not
  bool is_seqs();         // is MPI_COMM_SELF or not
  bool is_seqs_scalar();  // is seqs and scalar or not

  bool has_local_data() const;    // Array has local data or not

  // get & set Array's hash value
  void set_hash(const size_t &hash);
  size_t get_hash() const;

  // get & set Array's position
  void set_pos(int p);
  int get_pos();

  // set & get pseudo state
  void set_pseudo(bool ps);
  void set_pseudo();
  bool is_pseudo();

  // set & get bitset
  void set_bitset(std::string);
  void set_bitset(bitset<3> bs);
  void set_bitset();
  bitset<3> get_bitset();

  int get_stencil_width() const;    // get stencil width
  int get_stencil_type() const;     // get stencil type

  void set_zeros();     // set array's buffer = 0, easy for debug
  
  // when calculate with different dimensions, sometimes need to make pseudo 3d
  bool has_pseudo_3d();       // check has pseudo 3d or not
  void reset_pseudo_3d();     // reset pseudo 3d array
  ArrayPtr get_pseudo_3d();   // get pseudo 3d array
  void set_pseudo_3d(ArrayPtr ap);  // set pseudo 3d array as ap

  void reset();   // reset Array's attribute

  static void copy(ArrayPtr& dst, const ArrayPtr& src); // copy

  // these function is not used yet, 
  // if the time cost of update ghost is high,
  // we can change the logic of update ghost to do some optimization
  void update_lb_ghost_updated(int3 lb);
  void update_rb_ghost_updated(int3 rb);
  bool get_lb_ghost_updated(int dimension);
  bool get_rb_ghost_updated(int dimension);
  void reset_ghost_updated();

};



#endif
