/*
 * not used currently
 *
=======================================================*/

#include "Boundary.hpp"

using namespace std;

Boundary::Boundary(int size) {
  m_buffer = (void*) new double[size];
  m_size = size;
}

Boundary::~Boundary() {
  delete((double*) m_buffer);
}

void* Boundary::get_buffer() {
  return m_buffer;
}

void Boundary::set_size(int sz) {
  m_size = sz;
}

int Boundary::size() {
  return m_size;
}
