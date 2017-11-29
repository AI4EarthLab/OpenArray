#ifndef __C_OA_UTILS_HPP__
#define __C_OA_UTILS_HPP__

#include <mpi.h>

extern "C" {
  void oa_mpi_init();

  void oa_mpi_finalize();

  void get_rank(int* rank, MPI_Fint fcomm);
}

#endif