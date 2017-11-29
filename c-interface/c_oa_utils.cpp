#include "c_oa_utils.hpp"
#include <mpi.h>

extern "C" {
  void oa_mpi_init() {
    MPI_Init(NULL, NULL);
  }

  void oa_mpi_finalize() {
    MPI_Finalize();
  }

  void get_rank(int* rank, MPI_Fint fcomm) {
      MPI_Comm comm = MPI_Comm_f2c(fcomm);
      MPI_Comm_rank(comm, rank);
  }
}