#include "Array.hpp"
#include "mpi.h"
#include "common.hpp"
#include "utils/utils.hpp"

using namespace std;

Array::Array(const PartitionPtr &ptr, int data_type) : 
  m_data_type(data_type), m_par_ptr(ptr) {
	set_corners();
	Box box = get_corners();
	int sw = ptr->get_stencil_width();
	int size = box.size(sw);		
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
	
	// rank 0 recv & others send
	if(my_rank == 0){
		global_buf = new char[gs[0] * gs[1] * gs[2] * 
			oa::utils::data_size(m_data_type)];
		for(int z = 0; z < npz; ++z) {
			for(int y = 0; y < npy; ++y) {
	      for(int x = 0; x < npx; ++x) {
		
					Box box = m_par_ptr -> get_local_box({x, y, z});
					
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
							&reqs[target_rank]);
					
					MPI_Type_free(&target_sub_array);
				}
			}
		}

	}

	// all process send subarray to global_buf
	Box box = get_corners();
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

	if(my_rank == 0){
		MPI_Waitall(num_procs-1, &reqs[1], MPI_STATUSES_IGNORE);
		oa::utils::print_data((void*)global_buf, gs, m_data_type);
		delete(global_buf);
	}

}

// set local box in each process
void Array::set_corners() {
	m_corners = m_par_ptr->get_local_box();
}

// get local box in each process
Box Array::get_corners() {
	return m_corners;
}

// return box shape in each process
Shape Array::local_shape() {
	Box box = get_corners();
	return box.shape();
}

int Array::local_size() {
	Box box = get_corners();
	return box.size();
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

// set partition hash
void Array::set_hash(const size_t &hash) {
	m_hash = hash;  
}

// get partition hash
size_t Array::get_hash() const{
	return m_hash;
}
