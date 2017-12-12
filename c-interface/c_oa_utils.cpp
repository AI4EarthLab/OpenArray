#include "c_oa_utils.hpp"
//#include "../Init.hpp"
#include <mpi.h>

extern "C" {
  void c_get_rank(int* rank, MPI_Fint fcomm) {
      MPI_Comm comm = MPI_Comm_f2c(fcomm);
      MPI_Comm_rank(comm, rank);
  }

  void c_init(){
    //oa::init();
  }

  void c_finalize(){
    //oa::finalize();
  }
}
