#include "Array.hpp"
#include "mpi.h"
using namespace std;

Array :: Array(PartitionPtr ptr) : m_par_ptr(ptr) {}

Array :: Array(PartitionPtr ptr, void* data, int data_type) : 
    m_par_ptr(ptr), m_buffer(data), m_data_type(data_type) {}

int Array :: data_type() {
    return m_data_type;
}

void* Array :: buffer() {
    return m_buffer;
}

PartitionPtr Array :: partition() {
    return m_par_ptr;
}

void Array :: display(const char *prefix) {

}

// get local box in each process
BoxPtr Array :: corners() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return m_par_ptr->get_local_box(rank);
}

// return box shape in each process
vector<int> Array :: local_shape() {
    BoxPtr box_ptr = corners();
    return box_ptr->shape();
}

int Array :: local_size() {
    BoxPtr box_ptr = corners();
    return box_ptr->size();
}

// return global shape of Array
vector<int> Array :: shape() {
    return m_par_ptr->shape();
}

int Array :: size() {
    return m_par_ptr->size();
}
