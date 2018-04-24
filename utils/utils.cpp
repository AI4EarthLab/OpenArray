#include "utils.hpp"



namespace oa {
  namespace utils {

    int& get_disp_format(){
      static int flag = 1;
      return flag;
    }
    
    void set_disp_format(int flag){
      get_disp_format() = flag;      
    }
    
    //! display array for a buffer. 
    void print_data(void* buf, const Shape& shape, DATA_TYPE dt, int is, int ie, int js, int je, int ks, int ke) {
      switch(dt){
      case DATA_BOOL:
        print_data_t((bool*)buf, shape, is, ie, js, je, ks, ke);
        break;
      case DATA_INT:
        print_data_t((int*)buf, shape, is, ie, js, je, ks, ke);
        break;
      case DATA_FLOAT:
        print_data_t((float*)buf, shape, is, ie, js, je, ks, ke);
        break;
      case DATA_DOUBLE:
        print_data_t((double*)buf, shape, is, ie, js, je, ks, ke);
        break;
      }
    }

    DataType cast_data_type(DataType t1, DataType t2) {
      return std::max(t1, t2);
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

    bool is_equal_shape(const Shape& u, const Shape& v) {
      return u[0] == v[0] && u[1] == v[1] && u[2] == v[2];
    }

    std::string get_type_string(DataType t){
      switch(t){
      case DATA_INT:
        return "int";
        break;
      case DATA_FLOAT:
        return "float";
        break;
      case DATA_DOUBLE:
        return "double";
        break;
      case DATA_UNKNOWN:
        return "unknow";
        break;
      }
      return "unknow";
    }

    int get_shape_dimension(Shape S) {
      int d = 0;
      for (int i = 0; i < 3; i++) {
        if (S[i] > 1) d++;
      }
      return d;
    }

    bool check_legal_shape_calc(Shape u, Shape v) {
      for (int i = 0; i < 3; i++) {
        if (u[i] != v[i] && u[i] != 1 && v[i] != 1) return false;
      }
      return true;
    }

  }
}
