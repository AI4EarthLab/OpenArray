#include "c_oa_utils.hpp"
#include <mpi.h>

extern "C" {
  void oa_mpi_init() {
    MPI_Init(NULL, NULL);
  }

  void oa_mpi_finalize() {
    MPI_Finalize();
  }
}