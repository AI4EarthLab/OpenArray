/*
 * Init.cpp:
 *
=======================================================*/

#include "mpi.h"
#include "MPI.hpp"
#include "Init.hpp"
#include "Partition.hpp"
#ifdef __HAVE_CUDA__
  #include "CUDA.hpp"
#endif

namespace oa{
  // init the MPI, init Partition default information
  void init(int comm, Shape procs_shape,
          int argc, char** argv){
    oa::MPI::global()->init(comm, argc, argv);
    #ifdef __HAVE_CUDA__
      std:cout<<"Init gpu envirionment\n";
      MPI_Comm c_comm = MPI_Comm_f2c(comm); 
      oa::gpu::initialize_gpu(c_comm);
    #endif
    
    Partition::set_default_procs_shape(procs_shape);
    Partition::set_default_stencil_width(1);
  }

  // finalize the MPI
  void finalize(){
    oa::MPI::global()->finalize();
  }
}
