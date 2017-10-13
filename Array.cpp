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
