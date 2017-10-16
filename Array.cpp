#include "Array.hpp"
#include "mpi.h"
using namespace std;

Array::Array(const PartitionPtr &ptr, int data_type) : 
    m_par_ptr(ptr), m_data_type(data_type) {
    set_corners();
    BoxPtr box_ptr = get_corners();
    int sw = ptr->get_stencil_width();
    int size = box_ptr->size(sw);
    
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
    std::cout<<"Array destructor called!"<<std::endl;
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

const int Array::get_data_type() const{
    return m_data_type;
}

void* Array::get_buffer() {
    return m_buffer;
}

const PartitionPtr Array::get_partition() const{
    return m_par_ptr;
}

void Array::display(const char *prefix) {

  Shape ps = m_par_ptr->procs_shape();

  int npx = s[0];
  int npy = s[1];
  int npz = s[2];
  
  int num_procs = npx * npy * npz;

  Shape gs = shape();
  char* global_buf = NULL;
  
  MPI_Request reqs[num_procs];
  
  if(my_rank() == 0){
    global_buf = new char[gs[0] * gs[1] * gs[2] * DATA_SIZE(m_data_type)];
    for(int z = 0; z < npz; ++z){
      for(int y = 0; y < npy; ++y){
	for(int x = 0; x < npx; ++x){
	  
	  BoxPtr bp = m_par_ptr -> get_local_box({x, y, z});
	  
	  int xs, ys, zs, xe, ye, ze;
	  bp->get_corners(xs, xe, ys, ye, zs, ze);
	  
	  MPI_Datatype target_sub_array;
	  int starts[3] = {xs, ys, zs};
	  int sw = m_par_ptr -> get_stencil_width();
	  int subsize[3] = {xe-xs, ye-ys, ze-zs};
	  int bigsize[3] = {gs[0], gs[1], gs[2]};

	  MPI_Type_create_subarray(3, bigsize, subsize,
				   starts, MPI_ORDER_C,
				   oa::utils::MPI_Datatype(m_data_type),
				   &target_sub_array);
	  
	  MPI_Type_commit(&target_sub_array);
	  int target_rank = m_par_ptr->get_procs_rank({x, y, z});
	  if(target_rank != 0){
	    MPI_Irecv(global_buf, 1,
		      target_sub_array,
		      target_rank, 100,
		      MPI_COMM_WORLD,
		      &reqs[target_rank]);
	  }
	}
      }
    }
  }else{
    MPI_Datatype mysubarray;

    int sw = m_par_ptr -> get_stencil_width();    
    int starts[3]  = {sw, sw, sw};
    int bigsize[3] = {xe-xs+sw, ye-ys+sw, ze-zs+sw};
    int subsize[3] = {xe-xs, ye-ys, ze-zs};
    MPI_Type_create_subarray(3, bigsize, subsize,
			     starts, MPI_ORDER_C,
			     oa::utils::MPI_Datatype(m_data_type),
			     &mysubarray);
    MPI_Type_commit(&mysubarray);
    MPI_Send(local_buf, 1, mysubarray, 0, 100, MPI_COMM_WORLD);
  }

  if(rank == 0){
    MPI_Waitall(size-1, &reqs[1], MPI_STATUSES_IGNORE);
    oa::utils::print_data((void*)global_buf, gs, m_data_type);
    delete(global_buf);
  }
}

// set local box in each process
void Array::set_corners() {
    m_corners = m_par_ptr->get_local_box();
}

// get local box in each process
BoxPtr Array::get_corners() {
    return m_corners;
}

// return box shape in each process
Shape Array::local_shape() {
    BoxPtr box_ptr = get_corners();
    return box_ptr->shape();
}

int Array::local_size() {
    BoxPtr box_ptr = get_corners();
    return box_ptr->size();
}

// return global shape of Array
Shape Array::shape() {
    return m_par_ptr->shape();
}

int Array::size() {
    return m_par_ptr->size();
}

// set partition hash
void Array::set_hash(const size_t &hash) {
    m_hash = hash;    
}

// get partition hash
const size_t Array::get_hash() const{
    return m_hash;
}
