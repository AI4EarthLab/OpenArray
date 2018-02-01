#include "Array.hpp"
#include "mpi.h"
#include "common.hpp"
#include "utils/utils.hpp"
#include "ArrayPool.hpp"

using namespace std;


Array::Array(const PartitionPtr &ptr, int data_type) : 
  m_data_type(data_type), m_par_ptr(ptr) {
  set_local_box();
  Box box = get_local_box();
  int sw = ptr->get_stencil_width();
  int size_in = box.size();
  int size = box.size(sw);

  // if box.size() == 0, there is no need to contain stencil
  // if (size_in == 0) size = 0;
  
  switch (m_data_type) {
  case DATA_INT:
    m_buffer = (void*) new int[size];
    break;

  case DATA_FLOAT:
    m_buffer = (void*) new float[size];
    break;

  case DATA_DOUBLE:
    m_buffer = (void*) new double[size];
    break; 
  }
}

Array::~Array(){
  //std::cout<<"Array destructor called!"<<std::endl;
  switch (m_data_type) {
  case DATA_INT:
    delete((int*) m_buffer);
    break;

  case DATA_FLOAT:
    delete((float*) m_buffer);
    break;

  case DATA_DOUBLE:
    delete((double*) m_buffer);
    break; 
  }

}

int Array::get_data_type() const{
  return m_data_type;
}

void* Array::get_buffer() {
  return m_buffer;
}

PartitionPtr Array::get_partition() const{
  return m_par_ptr;
}

void Array::display(const char *prefix) {

  Shape ps = m_par_ptr->procs_shape();

  int npx = ps[0];
  int npy = ps[1];
  int npz = ps[2];
  int my_rank = rank();
  
  int num_procs = npx * npy * npz;

  Shape gs = shape();
  char* global_buf = NULL;
  
  MPI_Request reqs[num_procs];
  int reqs_cnt = 0;
  //printf("gs:%d, %d, %d\n", gs[0], gs[1], gs[2]);
  // rank 0 recv & others send
  if (my_rank == 0) {
    global_buf = new char[gs[0] * gs[1] * gs[2] * 
                          oa::utils::data_size(m_data_type)];
    for(int z = 0; z < npz; ++z) {
      for(int y = 0; y < npy; ++y) {
        for(int x = 0; x < npx; ++x) {
    
          Box box = m_par_ptr -> get_local_box({x, y, z});
          if (box.size() <= 0) continue;

          int xs, ys, zs, xe, ye, ze;
          box.get_corners(xs, xe, ys, ye, zs, ze);
          
          MPI_Datatype target_sub_array;
          int starts[3] = {xs, ys, zs};
          int subsize[3] = {xe-xs, ye-ys, ze-zs};
          int bigsize[3] = {gs[0], gs[1], gs[2]};
      
          MPI_Type_create_subarray(3, bigsize, subsize,
                                   starts, MPI_ORDER_FORTRAN,
                                   oa::utils::mpi_datatype(m_data_type),
                                   &target_sub_array);
          
          MPI_Type_commit(&target_sub_array);
          int target_rank = m_par_ptr->get_procs_rank({x, y, z});
          MPI_Irecv(global_buf, 1,
                    target_sub_array,
                    target_rank, 100,
                    m_par_ptr->get_comm(),
                    &reqs[reqs_cnt++]);
          
          MPI_Type_free(&target_sub_array);
        }
      }
    }

  }

  // all process send subarray (if size > 0) to global_buf
  Box box = get_local_box();
  if (box.size() > 0) {
    int xs, ys, zs, xe, ye, ze;
    box.get_corners(xs, xe, ys, ye, zs, ze);
    int sw = m_par_ptr -> get_stencil_width();  
    
    MPI_Datatype mysubarray;
    int starts[3]  = {sw, sw, sw};
    int bigsize[3] = {xe-xs+2*sw, ye-ys+2*sw, ze-zs+2*sw};
    int subsize[3] = {xe-xs, ye-ys, ze-zs};
    MPI_Type_create_subarray(3, bigsize, subsize,
                             starts, MPI_ORDER_FORTRAN,
                             oa::utils::mpi_datatype(m_data_type),
                             &mysubarray);
    MPI_Type_commit(&mysubarray);
    MPI_Send(m_buffer, 1, mysubarray, 0, 100, m_par_ptr->get_comm());
    MPI_Type_free(&mysubarray);
  }

  if (my_rank == 0){
    printf("\n%s\n", prefix);
    std::cout<<"\tdata type = "
             << oa::utils::get_type_string(m_data_type)
             << std::endl;

    std::cout<<"\tpos = "
             << pos
             << std::endl;

    std::cout<<"\tis_pseudo = "
             << m_is_pseudo
             << std::endl;
    
    std::cout<<"\tbitset = "
             << m_bs
             << std::endl;

    m_par_ptr->display(NULL, true);
    
    MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);
    oa::utils::print_data((void*)global_buf, gs, m_data_type);
    delete(global_buf);
    printf("\n");
  }
}

// set local box in each process
void Array::set_local_box() {
  m_corners = m_par_ptr->get_local_box();
}

// get local box in each process
Box Array::get_local_box() const{
  return m_corners;
}

Shape Array::buffer_shape() const{
  int sw = m_par_ptr->get_stencil_width();
  return m_corners.shape(sw);
}

int Array::buffer_size() const {
  int sw = m_par_ptr->get_stencil_width();
  return m_corners.size(sw);
}

// return box shape in each process
Shape Array::local_shape() {
  Box box = get_local_box();
  return box.shape();
}

int Array::local_size() const{
  Box box = get_local_box();
  return box.size();
}

Box Array::local_data_win() const{
  const int sw = m_par_ptr->get_stencil_width();
  const Shape bs = this->buffer_shape();
  return Box(sw, bs[0]-sw, sw, bs[1]-sw, sw, bs[2]-sw);
}

// return global shape of Array
Shape Array::shape() {
  return m_par_ptr->shape();
}

int Array::size() {
  return m_par_ptr->size();
}

int Array::rank() {
  return m_par_ptr->rank(); 
}

bool Array::is_scalar() {
  return (m_par_ptr->shape() == Shape({{1,1,1}}));
}

bool Array::is_seqs() {
  return (m_par_ptr->get_comm() == MPI_COMM_SELF);
}

bool Array::is_seqs_scalar() {
  return is_scalar() && is_seqs();
}

// void Array::set_seqs() {
//   m_is_seqs = true;
// }

// void Array::set_scalar() {
//   m_is_scalar = true;
// }

bool Array::has_local_data() const {
  return local_size() > 0;
}

// set partition hash
void Array::set_hash(const size_t &hash) {
  m_hash = hash;  
}

// get partition hash
size_t Array::get_hash() const{
  return m_hash;
}

void Array::set_pos(int p) {
  pos = p;
}

int Array::get_pos() {
  return pos;
}

void Array::set_pseudo(bool ps) {
  m_is_pseudo = ps;
}

bool Array::is_pseudo() {
  return m_is_pseudo;
}

void Array::set_bitset(string s) {
  m_bs = std::bitset<3> (s);
}

void Array::set_bitset(bitset<3> bs) {
  m_bs = bs;
}

// set_bitset based on global shape
void Array::set_bitset() {
  for (int i = 0; i < 3; i++) {
    if (m_par_ptr->shape()[i] != 1) m_bs[2 - i] = 1;
    else {
      m_bs[2 - i] = 0;
      m_is_pseudo = true;
    }
  }
}

bitset<3> Array::get_bitset() {
  return m_bs;
}

int Array::get_stencil_width() const{
  return m_par_ptr->get_stencil_width();
}
int Array::get_stencil_type() const{
  return m_par_ptr->get_stencil_type();
}

void Array::set_zeros(){
  switch(m_data_type){
  case DATA_INT:
    oa::internal::set_buffer_consts<int>
      ((int*)m_buffer, buffer_size(), 0);
    break;
  case DATA_FLOAT:
    oa::internal::set_buffer_consts<float>
      ((float*)m_buffer, buffer_size(), 0);
    break;
  case DATA_DOUBLE:
    oa::internal::set_buffer_consts<double>
      ((double*)m_buffer, buffer_size(), 0);
    break;
  }
}

bool Array::has_pseudo_3d() {
  return m_has_pseudo_3d;
}

void Array::reset() {
  //m_hash = 0;
  pos = -1;
  reset_pseudo_3d();
}

void Array::reset_pseudo_3d() {
  m_has_pseudo_3d = false;
  m_pseudo_3d = NULL;
}

ArrayPtr Array::get_pseudo_3d() {
  return m_pseudo_3d;
}

void Array::set_pseudo_3d(ArrayPtr ap) {
  m_pseudo_3d = ap;
  m_has_pseudo_3d = true;
}
