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

	}
}