#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Kernel.hpp"
#include "../op_define.hpp"
#include "../MPI.hpp"
#include <cmath>

extern "C" {
  void c_has_nan_or_inf(int* res, ArrayPtr*& ap, int skip_sw){
    switch((*ap)->get_data_type()){
      ///:for t in ['float', 'double']
    case DATA_${t.upper()}$:
      {
        const ${t}$* buf = (${t}$*)(*ap)->get_buffer();
        Shape s = (*ap)->buffer_shape();
        int sw = (*ap)->get_stencil_width();

        int offset = sw + skip_sw;
        for(int k = offset; k < s[2] - offset; ++k){
          for(int j = offset; j < s[1] - offset; ++j){
            for(int i = offset; i < s[0] - offset; ++i){
              if(std::isnan(buf[i + j * s[0] + k * s[0] * s[1]]) ||
                      std::isinf(buf[i + j * s[0] + k * s[0] * s[1]])){
                *res = 1;
                return;
              }
            }
          }
        }
        break;
      }
      ///:endfor
    }
    *res = 0;
  }
}
