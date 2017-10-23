#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <iostream>
#include <math.h>
#include <mpi.h>
#include <algorithm>
#include <type_traits>
#include <boost/format.hpp>
#include "../common.hpp"

namespace oa {
  namespace utils {
    template<class T>
    void print_data_t(T* buf, const Shape& shape) {
      const int M = shape[0];
      const int N = shape[1];
      const int P = shape[2];
      const int buf_size = M * N * P;

      std::string frt_str;
      if(std::is_same<T, int>::value){
        frt_str = "%10d";
      } else if (std::is_same<T, float>::value
        || std::is_same<T, double>::value) {
        frt_str = "%15.10f";

      } else if (std::is_same<T, bool>::value) {
        frt_str = "%4d";
      }

      static auto abs_compare = [](T a, T b){
        return abs(a) < abs(b);
      };

      T val = 1;

      if(std::is_same<T, float>::value ||
        std::is_same<T, double>::value) {
        T max = *std::max_element(buf, buf+buf_size, abs_compare);

        int factor = (int)log10(max);

        if(factor > 2){
          std::cout<< " * 1E"<<factor<<std::endl;
          val  = val / pow(10, factor);
        }

        if(factor <= -3){
          std::cout<< " * 1E"<<factor-1<<std::endl;
          val  = val / pow(10, factor-1);
        }

        //std::cout<<"min = "<< max << std::endl;
        //std::cout<<"factor = "<<factor << std::endl;
      }

      //for (int i = 0; i < 24; i++) std::cout<<buf[i]<<" "<<std::endl;

      for(int k = 0; k < P; ++k) {
        std::cout<<"[k = " << k << "]" << std::endl;
        for(int i = 0; i < M; ++i) {
          for(int j = 0; j < N; ++j) {
            std::cout<<boost::format(frt_str) % (buf[i + j * M + k * M * N] * val);
          }
          std::cout<<std::endl;
        }
      }
    }

    template <typename T>
    DataType to_type() {
      if (std::is_same<T, int>::value) return DATA_INT;
      if (std::is_same<T, float>::value) return DATA_FLOAT;
      if (std::is_same<T, double>::value) return DATA_DOUBLE;
    }

    //! display array for a buffer. 
    void print_data(void* buf, const Shape& shape, DATA_TYPE dt);

    int data_size(int data_type);
    
    MPI_Datatype mpi_datatype(int t);
    
    void mpi_order_start(MPI_Comm comm);

    void mpi_order_end(MPI_Comm comm);

  }  
}

#endif