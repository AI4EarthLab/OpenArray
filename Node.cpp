#include "Node.hpp"
#include "NodeDesc.hpp"
#include "Operator.hpp"

using namespace std;

Node::Node() {}

Node::Node(NodePtr u) {}

//Node(ArrayPtr array);
Node::~Node() {}

void Node::display(char const *prefix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    printf("%s \n", prefix);
    printf("    id : %d\n", id);
    const NodeDesc& nd = oa::ops::get_node_desc(m_type);    
    printf("  type : %s\n", nd.name.c_str());
    printf("  hash : %d\n", m_hash);

    printf(" input : \n");
    for(int i = 0; i < m_input.size(); ++i)
      printf("     %p : %s\n", m_input[i].get(),
	     oa::ops::get_node_desc(m_input[i]->type()).name.c_str());
  }

  if (m_type == TYPE_DATA) {
    if (!m_is_seqs) m_data->display(prefix);
    else if (!rank) m_data->display(prefix);
  }
}

int Node::input_size() const {
  return m_input.size();
}

NodePtr Node::input(int i) {
  return m_input.at(i);
}

NodePtr Node::output(int i) {
  return m_output.at(i);
}

void Node::add_input(int pos, NodePtr in) {
  m_input.insert(m_input.begin() + pos, in);
}

void Node::add_output(int pos, NodePtr out) {
  m_output.push_back(out);
}

void Node::clear_input(){
  m_input.clear();
}

void Node::clear_output(){
  m_output.clear();
}

void Node::set_type(NodeType type) {
  m_type = type;
}

void Node::set_data(const ArrayPtr& ptr) {
  m_data = ptr;
}

void Node::set_id(int _id) {
  id = _id;
}

int Node::get_id() const {
  return id;
}

size_t Node::hash() {
  return m_hash;
}

void Node::set_hash(size_t hash) {
  m_hash = hash;
}

NodeType Node::type() {
  return m_type;
}

ArrayPtr Node::get_data() {
  return m_data;
}

bool Node::has_data() {
  return m_data != NULL;
}

void Node::clear_data(){
  m_data = NULL;
}

void Node::reset() {
  clear_data();
  clear_input();
  m_type = TYPE_UNKNOWN;
  m_hash = 0;
  id = -1;
}

bool Node::is_scalar() const {
  return m_is_scalar;
}

bool Node::is_seqs() const {
  return m_is_seqs;
}

void Node::set_scalar(bool value) {
  m_is_scalar = value;
  //m_data->set_scalar();
}

void Node::set_seqs(bool value) {
  m_is_seqs = value;
  //m_data->set_seqs();
}

bool Node::is_seqs_scalar() const {
  return m_is_seqs && m_is_scalar;
}

void Node::set_depth(int child_depth) {
  m_depth = child_depth + 1;
}

void Node::set_depth(int l, int r) {
  m_depth = max(l, r) + 1;
}

int Node::get_depth() const {
  return m_depth;
}

void Node::set_shape(const Shape &s) {
  m_global_shape = s;
}

Shape Node::shape() {
  return m_global_shape;
}

void Node::set_data_type(int dt) {
  m_data_type = dt;
}

int Node::get_data_type() const {
  return m_data_type;
}
