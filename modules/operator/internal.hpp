
#ifndef __OP_INTERNAL_HPP__
#define __OP_INTERNAL_HPP__

#include "../../Function.hpp"
namespace oa{
  namespace internal{
#define calc_id(i,j,k,S) ((k)*(S[0])*(S[1])+(j)*(S[0])+(i))


    // crate kernel_dxc
    // A = dxc(U)

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / 1.0;
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxc_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i+1, j, k, S)] - b[calc_id(i-1, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dyc
    // A = dyc(U)

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / 1.0;
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyc_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }
      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j+1, k, S)] - b[calc_id(i, j-1, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dzc
    // A = dzc(U)

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / 1.0;
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzc_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }
      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k+1, S)] - b[calc_id(i, j, k-1, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_axb
    // A = axb(U)

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i - 1, j, k, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_axf
    // A = axf(U)

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void axf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i + 1, j, k, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_ayb
    // A = ayb(U)

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j - 1, k, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_ayf
    // A = ayf(U)

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void ayf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j + 1, k, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_azb
    // A = azb(U)

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k - 1, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_azf
    // A = azf(U)

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void azf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 0.5 * (b[calc_id(i, j, k, S)] + b[calc_id(i, j, k + 1, S)]) / 1;
          }
        }
      }


    }


    // crate kernel_dxb
    // A = dxb(U)

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw; i < sw + lbound[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i - 1, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dxf
    // A = dxf(U)

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dxf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = sw + sp[0] - rbound[0]; i < sw + sp[0]; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i + 1, j, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dyb
    // A = dyb(U)

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw; j < sw + lbound[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j - 1, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dyf
    // A = dyf(U)

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dyf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = 0; k < P; k++) {
        for (int j = sw + sp[1] - rbound[1]; j < sw + sp[1]; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j + 1, k, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dzb
    // A = dzb(U)

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzb_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw; k < sw + lbound[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k, S)] - b[calc_id(i, j, k - 1, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


    // crate kernel_dzf
    // A = dzf(U)

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_ooo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / 1.0;
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_ooo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / 1.0;
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_ooz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_ooz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_oyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_oyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_oyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_oyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(o, j, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xoo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xoo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xoz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xoz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, o, k, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xyo_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xyo_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, o, SG)];
          }
        }
      }


    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xyz_calc_inside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int o = sw;

      for (int k = sw + lbound[2]; k < sw + sp[2] - rbound[2]; k++) {
        for (int j = sw + lbound[1]; j < sw + sp[1] - rbound[1]; j++) {
          for (int i = sw + lbound[0]; i < sw + sp[0] - rbound[0]; i++) {
            
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
            
          }
        }
      }
    }

    template<typename T1, typename T2, typename T3>
    void dzf_with_grid_xyz_calc_outside(T1* ans, T2* b, T3* g, oa_int3 lbound, oa_int3 rbound, 
            int sw, Shape sp, Shape S, Shape SG) {
      int M = S[0];
      int N = S[1];
      int P = S[2];

      int o = sw;

      
      // update outside one surface (contains boundary, doesn't care)

      for (int k = sw + sp[2] - rbound[2]; k < sw + sp[2]; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            ans[calc_id(i, j, k, S)] = 1.0 * (b[calc_id(i, j, k + 1, S)] - b[calc_id(i, j, k, S)]) / g[calc_id(i, j, k, SG)];
          }
        }
      }


    }


  }
}

#endif
