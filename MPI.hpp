
#include <iostream>
#include "mpi.h"

class MPI{

  MPI_Comm m_comm;

public:

  MPI();

  void init(MPI_Comm comm, int argc, char** argv);

  void init(int comm, int argc, char** argv);
  
  void finalize();
  
  MPI_Comm& comm();

  int rank();

  int size();
  
  static MPI* global();
  
};
