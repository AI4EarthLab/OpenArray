
#ifndef __MPI_HPP__
#define __MPI_HPP__

#include <iostream>
#include "mpi.h"
#include <vector>

namespace oa{
  class MPI{

    MPI_Comm m_comm;

  public:

    MPI();

    void c_init(MPI_Comm comm, int argc, char** argv);

    void init(int comm, int argc, char** argv);
  
    void finalize();
  
    MPI_Comm& comm();

    int rank();

    int rank(MPI_Comm comm);

    int size();

    int size(MPI_Comm comm);

    static MPI* global();


    void order_start();
    void order_end();

    // void *OA_MPI_Wait_Func(void *arg);
    // void OA_MPI_Wait_Begin(vector<MPI_Request>  *mra, pthread_t * tid);
    // void OA_MPI_Wait_End(pthread_t * tid);
    static void *wait_func(void *arg);
    static void wait_begin(std::vector<MPI_Request>  *mra, pthread_t * tid);
    static void wait_end(pthread_t * tid);

  };
}
#endif
