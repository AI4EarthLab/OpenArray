/*
 * Diagnosis.cpp:
 *
=======================================================*/

#include "mpi.h"
#include "MPI.hpp"
#include "ArrayPool.hpp"

namespace oa{
  namespace diag{
    bool has_nan_or_inf(ArrayPtr& ap, int skip_sw){
      int flag = 0;
      
      switch((ap)->get_data_type()){
      case DATA_FLOAT:
      {
        const float* buf = (float*)(ap)->get_buffer();
        Shape s = (ap)->buffer_shape();
        int sw = (ap)->get_stencil_width();
        Shape as = ap->shape();
        int offset_x = (as[0] == 1) ? sw : sw + skip_sw;
        int offset_y = (as[1] == 1) ? sw : sw + skip_sw;
        int offset_z = (as[2] == 1) ? sw : sw + skip_sw;

        for(int k = offset_z; k < s[2] - offset_z; ++k){
          for(int j = offset_y; j < s[1] - offset_y; ++j){
            for(int i = offset_x; i < s[0] - offset_x; ++i){
              if(std::isnan(buf[i + j * s[0] + k * s[0] * s[1]]) ||
                      std::isinf(buf[i + j * s[0] + k * s[0] * s[1]])){
                flag = 1;
              }
            }
          }
        }
        break;
      }
      case DATA_DOUBLE:
      {
        const double* buf = (double*)(ap)->get_buffer();
        Shape s = (ap)->buffer_shape();
        int sw = (ap)->get_stencil_width();
        Shape as = ap->shape();
        int offset_x = (as[0] == 1) ? sw : sw + skip_sw;
        int offset_y = (as[1] == 1) ? sw : sw + skip_sw;
        int offset_z = (as[2] == 1) ? sw : sw + skip_sw;

        for(int k = offset_z; k < s[2] - offset_z; ++k){
          for(int j = offset_y; j < s[1] - offset_y; ++j){
            for(int i = offset_x; i < s[0] - offset_x; ++i){
              if(std::isnan(buf[i + j * s[0] + k * s[0] * s[1]]) ||
                      std::isinf(buf[i + j * s[0] + k * s[0] * s[1]])){
                flag = 1;
              }
            }
          }
        }
        break;
      }
      default:
        printf("unsupported datatype.\n");
        exit(0);
        break;
      }

      int global_flag = 0;
      MPI_Comm comm = ap->get_partition()->get_comm();
      MPI_Allreduce(&flag, &global_flag, 1, MPI_INT, MPI_SUM, comm);

      if(global_flag != 0){
        return true;        
      }else{
        return false;
      }
    }    
  }
}
