#include "fusion-kernel.hpp"

// MPI_COMM default is MPI_COMM_WORLD
#define COMM MPI_COMM_WORLD

// stencil_width default is 1
#define STENCIL_WIDTH 1
  

///:set kernel_file = "fusion-kernels"
///:if os.path.isfile(kernel_file)
///:set lines = io.open(kernel_file).read().split('\n')
///:for i in lines[:-1]
///:set line = i.split(' ')
///:set key = line[0]
///:set expr = line[1]
ArrayPtr kernel_${key}$(vector<ArrayPtr> &ops, const Shape& s, int dt) {
  ArrayPtr ap = ArrayPool::global()->get(COMM, s, STENCIL_WIDTH, dt);
  int size = ap->buffer_size();
  for (int i = 0; i < size; i++) {
    ap[i] = 0;
  }
  return ap;
}
///:endfor
///:endif
