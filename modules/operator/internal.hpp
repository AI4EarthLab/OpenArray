
#ifndef __OP_INTERNAL_HPP__
#define __OP_INTERNAL_HPP__

#include "../../Function.hpp"
namespace oa{
  namespace internal{
#define calc_id(i,j,k,S) ((k)*(S[0])*(S[1])+(j)*(S[0])+(i))
    ///:mute
    ///:include "kernel_type.fypp"
    ///:endmute

    ///:for k in FUNC
    ///:set name = k[1]
    ///:set func = k[2]

    // crate kernel_${name}$
    // A = ${name}$(U)

    ///:for i in GRID
    ///:set grid = i[2]
    ///:set g = i[3]
    template<typename T1, typename T2, typename T3>
    void ${name}$_${grid}$_calc_inside(T1* ans, T2* b, T3* g, int3 lbound, int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      ///:if name[0] == 'a'
      ///:set g = 1
      ///:endif
      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ${name}$_${grid}$_calc_outside(T1* ans, T2* b, T3* g, int3 lbound, int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      ///:if name[0] == 'a'
      ///:set g = 1
      ///:endif
      
      // update outside one surface (contains boundary, doesn't care)

      ///:if name[1:] in ['zb', 'zc']
      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['zf', 'zc']
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['yb', 'yc']
      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['yf', 'yc']
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['xb', 'xc']
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif
      ///:if name[1:] in ['xf', 'xc']
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = ${func}$ / ${g}$;
          }
        }
      }
      ///:endif


    }

    ///:endfor
    ///:endfor

  }
}

#endif
