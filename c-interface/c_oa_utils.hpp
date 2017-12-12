
#ifndef __C_OA_UTILS_HPP__
#define __C_OA_UTILS_HPP__

#include <mpi.h>

extern "C" {
  void c_get_rank(int* rank, MPI_Fint fcomm);
}

#endif
