#include "utils.hpp"

namespace oa {
	namespace utils {
		//! display array for a buffer. 
    void print_data(void* buf, const Shape& shape, DATA_TYPE dt) {
      switch(dt){
        case DATA_BOOL:
          print_data_t((bool*)buf, shape);
          break;
        case DATA_INT:
          print_data_t((int*)buf, shape);
          break;
        case DATA_FLOAT:
          print_data_t((float*)buf, shape);
          break;
        case DATA_DOUBLE:
          print_data_t((double*)buf, shape);
          break;
      }
    }

    int data_size(int data_type) {
      int ds[3] = {4, 4, 8};
      return ds[data_type];
    }

    MPI_Datatype mpi_datatype(int t) {
      switch(t) {
        case DATA_BOOL:
          return MPI_C_BOOL; break;
        case DATA_INT:
          return MPI_INT; break;
        case DATA_FLOAT:
          return MPI_FLOAT; break;
        case DATA_DOUBLE:
          return MPI_DOUBLE; break;
        default:
          return MPI_INT;
      }
    }

    void mpi_order_start(MPI_Comm comm) {
      int rank;
      MPI_Comm_rank(comm, &rank);
      for (int i = 0; i < rank; i++)
        MPI_Barrier(comm);
    }

    void mpi_order_end(MPI_Comm comm) {
      int size;
      int rank;
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &size);
      for (int i = rank; i < size; i++)
        MPI_Barrier(comm);
    }
	}
}