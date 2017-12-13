#include "mpi.h"
#include "Init.hpp"

namespace oa{
  void init(int argc, char** argv){
    if(argc > 0){
      MPI_Init(&argc, &argv);
    }else{
      MPI_Init(NULL, NULL);
    }
  }

  void finalize(){
    MPI_Finalize();
  }
}
