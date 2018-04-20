/*
 * Init.cpp:
 *
=======================================================*/

#include "mpi.h"
#include "MPI.hpp"
#include "Init.hpp"
#include "Partition.hpp"

namespace oa{
  // init the MPI, init Partition default information
  void init(int comm, Shape procs_shape,
          int argc, char** argv){
    oa::MPI::global()->init(comm, argc, argv);

    Partition::set_default_procs_shape(procs_shape);
    Partition::set_default_stencil_width(1);
  }

  // finalize the MPI
  void finalize(){
    oa::MPI::global()->finalize();
  }
}
