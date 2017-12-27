#include "mpi.h"
#include "Init.hpp"
#include "MPI.hpp"
#include "Partition.hpp"

namespace oa{
  void init(int comm, Shape procs_shape,
          int argc, char** argv){
    oa::MPI::global()->init(comm, argc, argv);

    Partition::set_default_procs_shape(procs_shape);
    Partition::set_default_stencil_width(1);
  }

  
  void finalize(){
    oa::MPI::global()->finalize();
  }
}
