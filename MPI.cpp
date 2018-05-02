/*
 * MPI.cpp
 *
=======================================================*/

#include "MPI.hpp"
#include "c-interface/c_oa_cache.hpp"

namespace oa{
  MPI::MPI(){
    m_comm = MPI_COMM_NULL;
  }

  void MPI::c_init(MPI_Comm comm, int argc, char** argv){
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
    // comm is a fortran MPI_Comm, use f2c to change into c MPI_Comm
    m_comm = MPI_Comm_f2c(comm);
  }

  void MPI::finalize(){
    c_clear_cache();  // clear cache used in FSET(CSET)
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

  int MPI::rank(MPI_Comm comm){
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  int MPI::size(){
    int size;
    MPI_Comm_size(m_comm, &size);
    return size;
  }

  int MPI::size(MPI_Comm comm){
    int size;
    MPI_Comm_size(comm, &size);
    return size;
  }

  MPI* MPI::global(){
    static MPI obj;
    return &obj;
  }

  void MPI::order_start() {
    int r = rank();
    for (int i = 0; i < r; i++)
      MPI_Barrier(m_comm);
  }

  void MPI::order_end() {
    int s = size();
    int r = rank();
    for (int i = r; i < s; i++)
      MPI_Barrier(m_comm);
  }


  void *MPI::wait_func(void *arg) {
    int i,j;
    vector<MPI_Request>  *mra = (vector<MPI_Request>  *)arg;
    for(i=0;i<(*mra).size();i++)
      {
        MPI_Wait(&(*mra)[i], MPI_STATUSES_IGNORE);
      }
    return NULL;
  }

  void MPI::wait_begin(vector<MPI_Request>  *mra, pthread_t * tid) {
    pthread_create(tid, NULL, wait_func, (void *)(mra));
  }

  void MPI::wait_end(pthread_t * tid) {
    pthread_join(*tid,NULL);
  }

}
