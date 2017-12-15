
#ifndef __OP_INTERNAL_HPP__
#define __OP_INTERNAL_HPP__

#include "../../Function.hpp"
namespace oa{
  namespace internal{
    ///:mute
    ///:include "../../NodeType.fypp"
    ///:endmute

    ///:for k in [i for i in L if i[3] == 'D']
    ///:set name = k[1]
    
    ///:set func = ''
    ///:if name == "axb"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)])'
    ///:elif name == "axf"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)])'
    ///:elif name == "ayb"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)])'
    ///:elif name == "ayf"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)])'
    ///:elif name == "azb"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)])'
    ///:elif name == "azf"
    ///:set func = '0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)])'
    ///:elif name == "dxb"
    ///:set func = '1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / 1.0'
    ///:elif name == "dxf"
    ///:set func = '1.0 * (-b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1.0'
    ///:elif name == "dyb"
    ///:set func = '1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / 1.0'
    ///:elif name == "dyf"
    ///:set func = '1.0 * (-b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1.0'
    ///:elif name == "dzb"
    ///:set func = '1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / 1.0'
    ///:elif name == "dzf"
    ///:set func = '1.0 * (-b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1.0'
    ///:elif name == "dxc"
    ///:set func = '0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)])'
    ///:elif name == "dyc"
    ///:set func = '0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)])'
    ///:elif name == "dzc"
    ///:set func = '0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)])'
    ///:endif

    // crate kernel_${name}$
    // A = ${name}$(U)
    template<typename T>
    void ${name}$_calc_inside(double* ans, T* b, int3 lbound, int3 rbound, 
            int sw, Shape sp, Shape S) {

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = ${func}$;
            
          }
        }
      }
    }

    template<typename T>
    void ${name}$_calc_outside(double* ans, T* b, int3 lbound, int3 rbound, 
            int sw, Shape sp, Shape S) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      // update outside one surface (contains boundary, doesn't care)

      ///:if name[1:] in ['zb', 'zc']
      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['zf', 'zc']
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['yb', 'yc']
      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['yf', 'yc']
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['xb', 'xc']
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['xf', 'xc']
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$;
          }
        }
      }
      ///:endif


    }

    ///:endfor

  }
}

#endif
