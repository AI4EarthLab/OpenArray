
#ifndef __SUM_INTERNAL_HPP__
#define __SUM_INTERNAL_HPP__

namespace oa{
  namespace internal{

    template<typename T>
      void buffer_sum_scalar_const(T *val, T *A, Box box, int sw, int size) {
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;
        *val = 0;

        for (int k = sw; k < K-sw; k++) {
          for (int j = sw; j < N-sw; j++) {
            for (int i = sw; i < M-sw; i++) {
              *val += A[k * M * N + j * M + i];
            }
          }
        }

      }

    template<typename T>
      void buffer_csum_x_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
#define t1 (M*N*k+M*j+i)
#define t2 (M*N*k+M*j+i+1) 
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        if(type == 2) 
          for(int i = 0; i < (N-2*sw)*(K-2*sw); i++)
            buffer[i] = 0;

        if(type == 1 || type == 0){
          int ndex = 0;
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                if((i == sw))
                  ap[t1] = buffer[(k-sw)*(N-sw-sw)+(j-sw)];
                else 
                  ap[t1] = 0;
              }
            }
          }
        }else{
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1] = 0;
              }
            }
          }
        }

        if(type == 1 || type == 2){
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1]+=A[t1];
                if(i < M - sw - 1)
                  ap[t2] += ap[t1];
                else 
                  buffer[(k-sw)*(N-sw-sw)+(j-sw)] = ap[t1];
              }
            }
          }
        }else{
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1]+=A[t1];
                if(i < M - sw - 1){
                  ap[t2] += ap[t1];
                }
              }
            }
          }
        }
#undef t1
#undef t2
      }


    template<typename T>
      void buffer_csum_y_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
#define t1  (M*N*k+M*j+i)
#define t2  (M*N*k+M*j+i+M)
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        if(type == 2) 
          for(int i = 0; i < (M-2*sw)*(K-2*sw); i++)
            buffer[i] = 0;

        if(type == 1 || type == 0){
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                if((j == sw))
                  ap[t1] = buffer[(k-sw)*(M-sw-sw)+(i-sw)];
                else 
                  ap[t1] = 0;
              }
            }
          }   
        }else{
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1] = 0;
              }
            }
          }   
        }

        if(type == 1 || type == 2){
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1]+=A[t1];
                if(j < N - sw -1){
                  ap[t2] += ap[t1];
                }
                if((j == N - sw -1) )
                  buffer[(k-sw)*(M-sw-sw)+(i-sw)] = ap[t1];
              }
            }
          }
        }else{
          for (int k = sw; k < K-sw; k++) {
            for (int j = sw; j < N-sw; j++) {
              for (int i = sw; i < M-sw; i++) {
                ap[t1]+=A[t1];
                if(j < N - sw -1){
                  ap[t2] += ap[t1];
                }
              }
            }
          }
        }
#undef t1
#undef t2
      }

    template<typename T>
      void buffer_csum_z_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
#define t1  (M*N*k+M*j+i)
#define t2  (M*N*k+M*j+i+M*N)
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        if(type == 2) 
          for(int i = 0; i < (N-2*sw)*(M-2*sw); i++)
            buffer[i] = 0;

        if(type == 1 || type == 0){
          for (int k = sw; k < K - sw; k++) {
            for (int j = sw; j < N - sw; j++) {
              for (int i = sw; i < M - sw; i++) {
                if((k == sw))
                  ap[t1] = buffer[(j-sw)*(M-sw-sw)+(i-sw)];
                else 
                  ap[t1] = 0;
              }
            }
          }
        }else{
          for (int k = sw; k < K - sw; k++) {
            for (int j = sw; j < N - sw; j++) {
              for (int i = sw; i < M - sw; i++) {
                ap[t1] = 0;
              }
            }
          }
        }

        if(type == 1 || type == 2){
          for (int k = sw; k < K - sw; k++) {
            for (int j = sw; j < N - sw; j++) {
              for (int i = sw; i < M - sw; i++) {
                ap[t1]+=A[t1];
                if(k < K - sw - 1){
                  ap[t2] += ap[t1];
                }
                if(k == K - sw - 1 )
                  buffer[(j-sw)*(M-sw-sw)+(i-sw)] = ap[t1];
              }
            }
          }
        }else{
          for (int k = sw; k < K - sw; k++) {
            for (int j = sw; j < N - sw; j++) {
              for (int i = sw; i < M - sw; i++) {
                ap[t1]+=A[t1];
                if(k < K - sw - 1){
                  ap[t2] += ap[t1];
                }
              }
            }
          }
        }
#undef t1
#undef t2
      }


    template<typename T>
      void buffer_sum_z_const(T *ap, T *A, Box box, int sw, int size) {//process is not splited in z direction
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        int cnt = 0;
        int dcnt = 0;

        for (int j = sw; j < N - sw; j++) {
#pragma simd
          for (int i = sw; i < M - sw; i++) {
            int temp1 = M*N*(K - sw - 1) + M*j + i;
            ap[temp1] = 0;
          }
        }
        for (int k = sw; k < K - sw; k++) {
          for (int j = sw; j < N - sw; j++) {
#pragma simd
            for (int i = sw; i < M - sw; i++) {
              int temp1 = M*N*(K - sw - 1) + M*j + i;
              int temp2 = M*N*k + M*j + i;
              ap[temp1] += A[temp2];
            }
          }
        }
      }

  }
}
#endif
