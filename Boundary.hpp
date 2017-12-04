#ifndef __BOUNDARY_HPP__
#define __BOUNDARY_HPP__

#include <memory>

class Boundary;
typedef shared_ptr<Boundary> BoundaryPtr;

class Boundary {
private:
  void* m_buffer;
  int m_size;

public:
  void* get_buffer();
  void set_size(int sz);
  int size();
};

#endif