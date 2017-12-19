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
    printf("  lbound : [%d %d %d]\n", m_lbound[0], m_lbound[1], m_lbound[2]);
    printf("  rbound : [%d %d %d]\n", m_rbound[0], m_rbound[1], m_rbound[2]);
    printf("  need_update : %s\n", m_update_boundary ? "True" : "False");
    printf("  grid pos : %d\n", pos);
    std::cout<<"\tis_pseudo = "
             << m_is_pseudo
             << std::endl;
    
    std::cout<<"\tbitset = "
             << m_bs
             << std::endl;

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

NodePtr& Node::input(int i) {
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

ArrayPtr& Node::get_data() {
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
  m_depth = 0;
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

int3 Node::get_lbound() {
  return m_lbound;
}

int3 Node::get_rbound() {
  return m_rbound;
}

void Node::set_lbound(int3 lb) {
  m_lbound = lb;
}

void Node::set_rbound(int3 rb) {
  m_rbound = rb;
}

void Node::set_lbound(int3 b1, int3 b2) {
  m_lbound[0] = max(b1[0], b2[0]);
  m_lbound[1] = max(b1[1], b2[1]);
  m_lbound[2] = max(b1[2], b2[2]);
}

void Node::set_rbound(int3 b1, int3 b2) {
  m_rbound[0] = max(b1[0], b2[0]);
  m_rbound[1] = max(b1[1], b2[1]);
  m_rbound[2] = max(b1[2], b2[2]);
}


void Node::set_update() {
  m_update_boundary = true;
}

bool Node::need_update() {
  return m_update_boundary;
}

void Node::set_pos(int p) {
  pos = p;
}

int Node::get_pos() {
  return pos;
}

void Node::set_pseudo(bool ps) {
  m_is_pseudo = ps;
}

bool Node::is_pseudo() {
  return m_is_pseudo;
}

void Node::set_bitset(string s) {
  m_bs = bitset<3>(s);
}

void Node::set_bitset(bitset<3> bs) {
  m_bs = bs;
}

bitset<3> Node::get_bitset() {
  return m_bs;
}

void Node::set_ref(const Box& b){
  m_ref = b;
}

Box& Node::get_ref(){
  return m_ref;
}

bool Node::is_ref() const{
  return m_data_type == TYPE_REF;
}

bool Node::is_ref_data() const{
  return (m_data_type == TYPE_REF) &&
    (m_input.at(0)->get_data_type()== TYPE_DATA);
}

ArrayPtr& Node::get_ref_data(){
  assert(is_ref_data());
  return m_input.at(0)->get_data();
}
