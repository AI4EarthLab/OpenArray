/*
 * Node.hpp
 * node is the basic item in the expression graph
 * including: data node & operation node
 *
=======================================================*/

#ifndef __NODE_HPP__
#define __NODE_HPP__
#include "Array.hpp"
#include "common.hpp"
#include <vector>
#include <memory>
#include <bitset>
#include <string>

class Node;

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodeList;

class Node {
private:
  bool m_is_seqs = false;     // is sequences or not
  bool m_is_scalar = false;   // is scalar or not
  bool m_is_pseudo = false;   // is pseudo or not
  bool m_update_boundary = false;   // need update boundary or not

  int id;           // node id
  int pos = -1;     // node position
  int m_depth = 0;  // node depth in expression graph
  int m_data_type;  // node data type

  NodeType m_type;    // node type
  ArrayPtr m_data;    // if it's data node, the data array pointer
  NodeList m_input;   // input, exp: A = B + C, B and C is the input of operation node +
  NodeList m_output;  // ouput, not used yet

  size_t m_hash;      // node hash value, to identify each node in node pool
  
  // left(lower) boundary, default is [0,0,0]
  int3 m_lbound = {{0, 0, 0}};    
  // right(upper) boundary, [0,0,1] means the upper z-dimension needs halo region data
  int3 m_rbound = {{0, 0, 0}};    

  Shape m_global_shape = {{1, 1, 1}};   // node global shape, to do deduction in expression graph
  std::bitset<3> m_bs = std::bitset<3>(7);  // node bitset

  Box m_ref;    // if it's reference data, should have a reference box
  int m_slice;
  
  int m_data_list_size = 0;   // the size of data list in the fusion kernel

public:
  // Constructor
  Node();
  Node(NodePtr u);
  // Destructor
  ~Node();
  
  // display information for node
  void display(char const *prefix = "");

  int input_size() const;   // node's input size
  
  NodePtr& input(int i);    // node's input[i]

  NodePtr output(int i);    // node's output[i]

  void add_input(int pos, NodePtr in);  // add in to input[pos]

  void add_output(int pos, NodePtr out);  // add out to output[pos]

  void clear_input();   // clear node's input

  void clear_output();  // clear node's output

  void set_type(NodeType type);   // set node type

  NodeType type();                // get node type

  void set_id(int _id);   // set node id

  int get_id() const;     // get node id
  
  size_t hash();          // get node hash

  void set_hash(size_t);  // set node hash

  void set_data(const ArrayPtr& ptr); // set node data if it's data node

  ArrayPtr& get_data();   // get data node's data array

  bool has_data();      // check if has data or not

  void clear_data();    // clear data
  
  void reset();   // reset node

  bool is_scalar() const;   // is scalar node or not

  void set_scalar(bool value=true);   // set scalar

  bool is_seqs() const;   // is sequences or not

  void set_seqs(bool value=true);   // set sequences

  bool is_seqs_scalar() const;    // is sequences scalar or not

  void set_depth(int d);  // set depth if node has one child

  void set_depth(int left_child, int right_child);    // set depth if node has two children

  int get_depth() const;  // get node depth

  // set & get node shape
  void set_shape(const Shape &s);
  Shape shape();

  // set & get node's data type
  void set_data_type(int dt);
  int get_data_type() const;

  // get lbound/rbound
  int3 get_lbound();
  int3 get_rbound();

  // set lbound/rbound
  void set_lbound(int3 lb);
  void set_rbound(int3 rb);

  // set lbound/rbound when has two children
  void set_lbound(int3 left_lb, int3 right_lb);
  void set_rbound(int3 left_rb, int3 ribht_rb);

  // set & get update state
  void set_update(bool flag = true);
  bool need_update();

  // set & get node position
  void set_pos(int p);
  int get_pos();

  // set & get node is pseudo 3d or not
  void set_pseudo(bool ps);
  bool is_pseudo();

  // set & get node's bitset
  void set_bitset(string s);
  void set_bitset();
  void set_bitset(bitset<3> bs);
  bitset<3> get_bitset();

  // set & get ref box
  void set_ref(const Box& b);
  Box& get_ref();

  bool is_ref() const;  //check is reference type or not

  bool is_ref_data() const; // check has reference data or not

  ArrayPtr& get_ref_data(); // return reference data

  void set_slice(int k);

  int get_slice();

  int get_data_list_size();

  void set_data_list_size(int x);

};


#endif
