#ifndef __ARRAY_HPP__
#define __ARRAY_HPP__

#include <vector>
#include <memory>
#include <bitset>
#include <string>
#include "Partition.hpp"
#include "Internal.hpp"
/*
 * Array:
 *  buffer:   data
 *  PartitionPtr:   partition information
 *  BoxPtr:   Array can be a reference one 
 */
class Array;
typedef shared_ptr<Array> ArrayPtr;
class Grid;
typedef shared_ptr<Grid> GridPtr;

class Array {
  private:
  void *m_buffer;
  // bool m_is_field = false;
  // int m_grid_pos = -1;
  int m_data_type = 2;
  PartitionPtr m_par_ptr;
  BoxPtr m_ref_box_ptr;
  Box m_corners;
  bool m_is_scalar = false;
  bool m_is_seqs = false;
  size_t m_hash;
  int pos = -1;
  //GridPtr grid;
  bool m_is_pseudo = false;
  std::bitset<3> m_bs = std::bitset<3>(std::string("111"));
  ArrayPtr m_pseudo_3d;

  public:
  Array(const PartitionPtr &ptr, int data_type = DATA_DOUBLE); 
  ~Array();
  int get_data_type() const;
  void* get_buffer();
  void set_buffer(void *buffer, int size);
  PartitionPtr get_partition() const;
  void display(const char *prefix = "");
  void set_local_box();
  Box get_local_box() const;
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
  // void set_scalar();
  // void set_seqs();
  bool has_local_data() const;
  void set_hash(const size_t &hash);
  size_t get_hash() const;
  void set_pos(int p);
  int get_pos();
  void set_pseudo(bool ps);
  bool is_pseudo();
  void set_bitset(std::string);
  bitset<3> get_bitset();
  void set_bitset(bitset<3> bs);
  void set_bitset();
  int get_stencil_width() const;
  int get_stencil_type() const;
  void set_zeros();
  Box local_data_win() const;
  ArrayPtr get_pseudo_3d();
};



#endif
