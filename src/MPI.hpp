/*
 * MPI.hpp
 * high-level interface of some useful MPI function
 *
=======================================================*/

#ifndef __MPI_HPP__
#define __MPI_HPP__

#include "mpi.h"
#include <vector>
#include <iostream>

namespace oa{
  class MPI{

  MPI_Comm m_comm;

  public:

    // Constructor
    MPI();
    
    // init mpi
    void init(int comm, int argc, char** argv);

    // init for c-interface
    void c_init(MPI_Comm comm, int argc, char** argv);
  
    // mpi finalize
    void finalize();
  
    // get mpi_comm
    MPI_Comm& comm();

    // get mpi rank use default mpi_comm
    int rank();

    // get mpi rank use comm
    int rank(MPI_Comm comm);

    // get mpi size use default mpi_comm
    int size();

    // get mpi use comm
    int size(MPI_Comm comm);

    // static MPI class
    static MPI* global();

    // use order_start & order_end to debug
    void order_start();
    void order_end();

    // use pthread to support asynchronous mpi communication
    // this feature is not used right now
    static void *wait_func(void *arg);
    static void wait_begin(std::vector<MPI_Request>  *mra, pthread_t * tid);
    static void wait_end(pthread_t * tid);

  };
}
#endif
