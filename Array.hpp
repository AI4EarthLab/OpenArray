/*
 * Array:
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

  void set_local_box();
  Box get_local_box() const;

  int get_data_type() const;
  void* get_buffer();
  void set_buffer(void *buffer, int size);
  PartitionPtr get_partition() const;
  void display(const char *prefix = "");
  Shape buffer_shape() const;
  int buffer_size() const;
  Shape local_shape();
  int local_size() const;
  Shape shape();
  int size();
  int rank();
  bool is_scalar();
  bool is_seqs();
  bool is_seqs_scalar();
  bool has_local_data() const;
  void set_hash(const size_t &hash);
  size_t get_hash() const;
  void set_pos(int p);
  int get_pos();
  void set_pseudo(bool ps);
  void set_pseudo();
  bool is_pseudo();
  void set_bitset(std::string);
  bitset<3> get_bitset();
  void set_bitset(bitset<3> bs);
  void set_bitset();
  int get_stencil_width() const;
  int get_stencil_type() const;
  void set_zeros();
  Box local_data_win() const;
  bool has_pseudo_3d();
  void reset_pseudo_3d();
  ArrayPtr get_pseudo_3d();
  void set_pseudo_3d(ArrayPtr ap);
  void reset();
  void update_lb_ghost_updated(int3 lb);
  void update_rb_ghost_updated(int3 rb);
  bool get_lb_ghost_updated(int dimension);
  bool get_rb_ghost_updated(int dimension);
  void reset_ghost_updated();

  static void copy(ArrayPtr& dst, const ArrayPtr& src);
};



#endif
