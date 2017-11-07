
#ifndef __NODE_HPP__
#define __NODE_HPP__
#include "Array.hpp"
#include "common.hpp"
#include <vector>
#include <memory>

class Node;

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodeList;

class Node {
private:
  int id;
  ArrayPtr m_data;
  NodeList m_input;
  NodeList m_output;
  size_t m_hash;
  NodeType m_type;
  bool m_is_seqs = false;
  bool m_is_scalar = false;
  //BoxPtr ref;

public:
  Node();

  Node(NodePtr u);
  //Node(ArrayPtr array);
  ~Node();
  
  void display(char const *prefix = "");

  int input_size() const;
  
  NodePtr input(int i);

  NodePtr output(int i);

  void add_input(int pos, NodePtr in);

  void add_output(int pos, NodePtr out);

  void set_type(NodeType type);

  void set_data(const ArrayPtr& ptr);

  void set_id(int _id);

  int get_id() const;
  
  size_t hash();

  NodeType type();

  ArrayPtr get_data();

  bool has_data();

  void reset();

  void clear_input();

  void clear_output();

  void clear_data();
  
  // MPI_Comm_world [1x1x1]
  bool is_scalar() const;

  void set_scalar();

  // MPI_Comm_self 
  bool is_seqs() const;

  void set_seqs();

  // MPI_Comm_self & [1x1x1]
  bool is_seqs_scalar() const;

};


#endif
