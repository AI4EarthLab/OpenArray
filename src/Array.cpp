/*
 * Array.cpp
 * 
 *
=======================================================*/

#include "mpi.h"
#include "Array.hpp"
#include "common.hpp"
#include "Function.hpp"
#include "ArrayPool.hpp"
#include "utils/utils.hpp"

using namespace std;

#ifdef __HAVE_CUDA__
Array::Array(const PartitionPtr &ptr, int data_type, DeviceType dev_type) : 
  m_data_type(data_type), m_par_ptr(ptr), m_device_type(dev_type) {

  set_local_box();
  Box box = get_local_box();
  int sw = ptr->get_stencil_width();
  int size = box.size_with_stencil(sw);
  m_buffer = NULL;
  m_buffer_gpu = NULL; 
  m_device_type = dev_type;
  // malloc data buffer
  if (m_device_type == CPU){
    switch (m_data_type)
    {
    case DATA_INT:
      m_buffer = (void *)new int[size];
      break;

    case DATA_FLOAT:
      m_buffer = (void *)new float[size];
      break;

    case DATA_DOUBLE:
      m_buffer = (void *)new double[size];
      break;
    }
    m_newest_buffer = CPU;
  }
  else if(m_device_type == GPU){
    switch (m_data_type)
    {
    case DATA_INT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size * sizeof(int)));
      break;

    case DATA_FLOAT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size * sizeof(float)));
      break;

    case DATA_DOUBLE:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size * sizeof(double)));
      break;
    }
    m_newest_buffer = GPU;

  }
  else if(m_device_type == CPU_AND_GPU){
    switch (m_data_type)
    {
    case DATA_INT:
      m_buffer = (void *)new int[size];
      break;

    case DATA_FLOAT:
      m_buffer = (void *)new float[size];
      break;

    case DATA_DOUBLE:
      m_buffer = (void *)new double[size];
      break;
    }

    switch (m_data_type)
    {
    case DATA_INT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(int)));
      break;

    case DATA_FLOAT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(float)));
      break;

    case DATA_DOUBLE:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(double)));
      break;
    }
    m_newest_buffer = GPU;
 
  }
  else {
    std::cout<<"Get wrong device type when creating array object!\n";
    std::cout<<"Device type: "<<m_device_type<<endl;
    exit(EXIT_FAILURE);
  }
}

#else
Array::Array(const PartitionPtr &ptr, int data_type, DeviceType dev_type) : 
  m_data_type(data_type), m_par_ptr(ptr), m_device_type(CPU) {

  set_local_box();
  Box box = get_local_box();
  int sw = ptr->get_stencil_width();
  int size = box.size_with_stencil(sw);
  m_buffer = NULL;
  m_buffer_gpu = NULL; 
  m_newest_buffer = CPU;
   // malloc data buffer
  switch (m_data_type)
  {
  case DATA_INT:
    m_buffer = (void *)new int[size];
    break;

  case DATA_FLOAT:
    m_buffer = (void *)new float[size];
    break;

  case DATA_DOUBLE:
    m_buffer = (void *)new double[size];
    break;
  }
}

#endif

Array::~Array(){
  // std::cout<<"Array destructor called!"<<std::endl;
  // delete data buffer

#ifndef __HAVE_CUDA__ 
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
#else
  if (m_device_type == CPU)
  {
    switch (m_data_type)
    {
    case DATA_INT:
      delete ((int *)m_buffer);
      break;

    case DATA_FLOAT:
      delete ((float *)m_buffer);
      break;

    case DATA_DOUBLE:
      delete ((double *)m_buffer);
      break;
    }
  }
  else if (m_device_type == GPU){
       CUDA_CHECK(cudaFree(m_buffer_gpu));
  }
  
  else if(m_device_type == CPU_AND_GPU){
     switch (m_data_type)
    {
    case DATA_INT:
      delete ((int *)m_buffer);
      break;

    case DATA_FLOAT:
      delete ((float *)m_buffer);
      break;

    case DATA_DOUBLE:
      delete ((double *)m_buffer);
      break;
    }
    cudaFree(m_buffer_gpu);
  }

  else {
    std::cout<<"Get wrong device type when destroying array object!\n";
    std::cout<<"Device type: "<<m_device_type<<endl;
    exit(EXIT_FAILURE);
  }
  m_buffer_gpu = NULL;
#endif
  m_buffer = NULL;
}

int Array::get_data_type() const{
  return m_data_type;
}

void* Array::get_buffer(DeviceType dt) {
  #ifndef __HAVE_CUDA__
    return m_buffer;
  #else
    if(m_newest_buffer == CPU)
    {
      if(dt != CPU) std::cout<<"warning: get_buffer from CPU\n"<<endl;
      return m_buffer;
    }
    return m_buffer_gpu;
  #endif
}

PartitionPtr Array::get_partition() const{
  return m_par_ptr;
}

void Array::display_info(const char *prefix) {
  if (rank() == 0){
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
    
    printf("\n");
  }
}

void Array::display(const char *prefix, int is, int ie, int js, int je, int ks, int ke) {

  Shape ps = m_par_ptr->procs_shape();

  int npx = ps[0];
  int npy = ps[1];
  int npz = ps[2];
  int my_rank = rank();
  
  int num_procs = npx * npy * npz;
  #ifdef __HAVE_CUDA__
    if(m_newest_buffer == GPU){
	memcopy_gpu_to_cpu();
	
     }
  #endif

  Shape gs = shape();
  char* global_buf = NULL;
  
  MPI_Request reqs[num_procs];
  int reqs_cnt = 0;

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
    oa::utils::print_data((void*)global_buf, gs, m_data_type, is, ie, js, je, ks, ke);
    delete(global_buf);
    printf("\n");
  }
}

void Array::set_local_box() {
  m_corners = m_par_ptr->get_local_box();
}

Box Array::get_local_box() const{
  return m_corners;
}

Shape Array::buffer_shape() const{
  int sw = m_par_ptr->get_stencil_width();
  return m_corners.shape_with_stencil(sw);
}

int Array::buffer_size() const {
  int sw = m_par_ptr->get_stencil_width();
  return m_corners.size_with_stencil(sw);
}

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

bool Array::has_local_data() const {
  return local_size() > 0;
}

void Array::set_hash(const size_t &hash) {
  m_hash = hash;  
}

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

// set pseudo automatically based on array's shape and procs_shape
void Array::set_pseudo() {
  for (int i = 0; i < 3; i++) {
    // array's shape [m,n,1], process's shape [px,py,1]  pseudo = false
    // array's shape [m,n,1], process's shape [px,py,pz(>1)], pseudo = true
    if (m_par_ptr->shape()[i] == 1 && m_par_ptr->procs_shape()[i] != 1) {
      m_is_pseudo = true;
    }
  }
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

// set_bitset automatically based on global shape
void Array::set_bitset() {
  for (int i = 0; i < 3; i++) {
    if (m_par_ptr->shape()[i] != 1) m_bs[2 - i] = 1;
    else {
      m_bs[2 - i] = 0;
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
  #ifndef __HAVE_CUDA__
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
  #else 
  switch(m_data_type){
  case DATA_INT:
    oa::gpu::set_buffer_consts<int>
      ((int*)m_buffer_gpu, buffer_size(), 0);
    break;
  case DATA_FLOAT:
    oa::gpu::set_buffer_consts<float>
      ((float*)m_buffer_gpu, buffer_size(), 0);
    break;
  case DATA_DOUBLE:
    oa::gpu::set_buffer_consts<double>
      ((double*)m_buffer_gpu, buffer_size(), 0);
    break;
  }
  #endif

}

bool Array::has_pseudo_3d() {
  return m_has_pseudo_3d;
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

void Array::reset() {
  pos = -1;
  reset_pseudo_3d();
  // reset_ghost_updated(); // not used yet
}

void Array::copy(ArrayPtr& dst, const ArrayPtr& src){
  //same datatype, same stencil width
  if(dst->get_hash() == src->get_hash()){

    dst->set_pos(src->get_pos());
    dst->set_pseudo(src->is_pseudo());
    dst->set_bitset(src->get_bitset());
    dst->reset_pseudo_3d();
    
    DataType dt = src->get_data_type();
    
    #ifndef __HAVE_CUDA__
    switch(dt) {
    case DATA_INT:
      oa::internal::copy_buffer(
          (int*) (dst->get_buffer()),
          (int*) (src->get_buffer()),
          dst->buffer_size());
      break;
    case DATA_FLOAT:
      oa::internal::copy_buffer(
          (float*) (dst->get_buffer()),
          (float*) (src->get_buffer()),
          dst->buffer_size());
      break;
    case DATA_DOUBLE:
      oa::internal::copy_buffer(
          (double*) (dst->get_buffer()),
          (double*) (src->get_buffer()),
          dst->buffer_size());
      break;
    }
    #else
    switch(dt) {
    case DATA_INT:
      oa::gpu::copy_buffer(
          (int*) (dst->get_buffer()),
          (int*) (src->get_buffer()),
          dst->buffer_size());
      break;
    case DATA_FLOAT:
      oa::gpu::copy_buffer(
          (float*) (dst->get_buffer()),
          (float*) (src->get_buffer()),
          dst->buffer_size());
      break;
    case DATA_DOUBLE:
      oa::gpu::copy_buffer(
          (double*) (dst->get_buffer()),
          (double*) (src->get_buffer()),
          dst->buffer_size());
      break;
    }
    #endif
 
  }else{
    // the partition is not same, need to transfer
    ArrayPtr dst1 = oa::funcs::transfer(src, dst->get_partition());
    dst = dst1;
  }
}

void Array::update_lb_ghost_updated(oa_int3 lb) {
  m_lb_ghost_updated[0] = m_lb_ghost_updated[0] || lb[0];
  m_lb_ghost_updated[1] = m_lb_ghost_updated[1] || lb[1];
  m_lb_ghost_updated[2] = m_lb_ghost_updated[2] || lb[2];
}

void Array::update_rb_ghost_updated(oa_int3 rb) {
  m_rb_ghost_updated[0] = m_rb_ghost_updated[0] || rb[0];
  m_rb_ghost_updated[1] = m_rb_ghost_updated[1] || rb[1];
  m_rb_ghost_updated[2] = m_rb_ghost_updated[2] || rb[2];
}

bool Array::get_lb_ghost_updated(int dimension) {
  return m_lb_ghost_updated[dimension];
}

bool Array::get_rb_ghost_updated(int dimension) {
  return m_rb_ghost_updated[dimension];
}

void Array::reset_ghost_updated() {
  m_lb_ghost_updated[0] = false;
  m_lb_ghost_updated[1] = false;
  m_lb_ghost_updated[2] = false;
  m_rb_ghost_updated[0] = false;
  m_rb_ghost_updated[1] = false;
  m_rb_ghost_updated[2] = false;
}

size_t gen_array_hash(size_t par_hash, int data_type, DeviceType dev_type){
    size_t hash = 0;
    hash = hash * 13131 + par_hash;
    hash = hash * 13131 + data_type;
    hash = hash * 13131 + dev_type;
    return hash;
  }


#ifdef __HAVE_CUDA__
void Array::update_hash(){
  size_t par_hash = m_par_ptr->get_hash();
  m_hash = gen_array_hash(par_hash, m_data_type, m_device_type);
}

void Array::memcopy_cpu_to_gpu(){
  if(m_buffer == NULL){
    std::cout<<"Get cpu memory buffer zero when copy memory from cpu to gpu!\n";
    exit(EXIT_FAILURE);
  }

  Box box = get_local_box();
  int sw = m_par_ptr->get_stencil_width();
  int size = box.size_with_stencil(sw);
  m_device_type = CPU_AND_GPU;
  update_hash();
  m_newest_buffer = GPU;
  if(m_buffer_gpu == NULL){
   switch(m_data_type){
    case DATA_INT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(int)));
      break;

    case DATA_FLOAT:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(float)));
      break;

    case DATA_DOUBLE:
      CUDA_CHECK(cudaMalloc((void **)&m_buffer_gpu, size*sizeof(double)));
      break;
    }
  }
  switch (m_data_type)
  {
  case DATA_INT:
    CUDA_CHECK(cudaMemcpy(m_buffer_gpu, m_buffer, size * sizeof(int), cudaMemcpyHostToDevice));
    break;

  case DATA_FLOAT:
    CUDA_CHECK(cudaMemcpy(m_buffer_gpu, m_buffer, size * sizeof(float), cudaMemcpyHostToDevice));
    break;

  case DATA_DOUBLE:
    CUDA_CHECK(cudaMemcpy(m_buffer_gpu, m_buffer, size * sizeof(double), cudaMemcpyHostToDevice));
    break;
  }
  return;
}


void Array::memcopy_gpu_to_cpu(){
  if(m_buffer_gpu == NULL) {
    std::cout<<"Get gpu memory buffer zero when copy memory from gpu to cpu!\n";
    exit(EXIT_FAILURE);
  }

  Box box = get_local_box();
  int sw = m_par_ptr->get_stencil_width();
  int size = box.size_with_stencil(sw);
  m_device_type = CPU_AND_GPU;
  update_hash();
  if(m_buffer == NULL){
    switch (m_data_type)
    {
    case DATA_INT:
      m_buffer = (void *)new int[size];
      break;

    case DATA_FLOAT:
      m_buffer = (void *)new float[size];
      break;

    case DATA_DOUBLE:
      m_buffer = (void *)new double[size];
      break;
    }
  }

  switch (m_data_type)
  {
  case DATA_INT:
    CUDA_CHECK(cudaMemcpy(m_buffer, m_buffer_gpu, size * sizeof(int), cudaMemcpyDeviceToHost));
    break;

  case DATA_FLOAT:
    CUDA_CHECK(cudaMemcpy(m_buffer, m_buffer_gpu, size * sizeof(float), cudaMemcpyDeviceToHost));
    break;

  case DATA_DOUBLE:
    CUDA_CHECK(cudaMemcpy(m_buffer, m_buffer_gpu, size * sizeof(double), cudaMemcpyDeviceToHost));
    break;
  }
  return;
}

void Array::set_newest_buffer(DeviceType dev_type){
  m_newest_buffer = dev_type;
}

void* Array::get_cpu_buffer(){
//if(flag == 1)
//{
//    if(m_newest_buffer == GPU) 
//        memcopy_gpu_to_cpu();
//}
    return m_buffer;
}

void* Array::get_gpu_buffer(){
//    if(m_newest_buffer == CPU)
//        memcopy_cpu_to_gpu();
    return m_buffer_gpu;

}
#else
void* Array::get_cpu_buffer(){
    return m_buffer;
}

#endif //#ifdef __HAVE_CUDA__
