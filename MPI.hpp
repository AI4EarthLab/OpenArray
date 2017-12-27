
#ifndef __MPI_HPP__
#define __MPI_HPP__

#include <iostream>
#include "mpi.h"

namespace oa{
  class MPI{

    MPI_Comm m_comm;

  public:

    MPI();

    void init(MPI_Comm comm, int argc, char** argv);

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
  };
}
#endif
