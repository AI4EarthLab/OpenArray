
#include "MPI.hpp"

MPI::MPI(){
  m_comm = MPI_COMM_NULL;
}

void MPI::init(MPI_Comm comm, int argc, char** argv){
  if(argc > 0){
    MPI_Init(&argc, &argv);
  }else{
    MPI_Init(NULL, NULL);
  }
  m_comm = comm;
}

void MPI::init(int comm, int argc, char** argv){
  if(argc > 0){
    MPI_Init(&argc, &argv);
  }else{
    MPI_Init(NULL, NULL);
  }
  m_comm = MPI_Comm_f2c(comm);
}

void MPI::finalize(){
  MPI_Finalize();
}
  
MPI_Comm& MPI::comm(){
  return m_comm;
}

int MPI::rank(){
  int rank;
  MPI_Comm_rank(m_comm, &rank);
  return rank;
}

int MPI::size(){
  int size;
  MPI_Comm_size(m_comm, &size);
  return size;
}
  
MPI* MPI::global(){
  static MPI obj;
  return &obj;
}

