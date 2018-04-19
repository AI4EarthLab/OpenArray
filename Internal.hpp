#ifndef __INTERNAL_HPP__
#define __INTERNAL_HPP__

#include <random>
#include "common.hpp"
#include "Box.hpp"
#include "utils/utils.hpp"
#include <vector>

// extern "C"{
//   void tic(const char* s);
//   void toc(const char* s);
//   void show_time(const char* s);
//   void show_all();
// }

using namespace std;

namespace oa {
  namespace internal {

    //int calc_id(int i, int j, int k, int3 S);

    template <typename T>
    void set_buffer_consts(T *buffer, int size, T val) {
      for (int i = 0; i < size; i++) buffer[i] = val;
    }
    
    template <typename T>
    void set_buffer_local(T* sub_buffer, const Box& box, int x, int y, int z, T val, int sw) {
      Shape sp = box.shape_with_stencil(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];

      int cnt = (z + sw) * M * N + (y + sw) * M + x + sw;
      sub_buffer[cnt] = val;
    }

    template <typename T>
    T get_buffer_local_sub(T* sub_buffer, const Box& box, int x, int y, int z, int sw) {
      Shape sp = box.shape_with_stencil(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];

      int cnt = (z + sw) * M * N + (y + sw) * M + x + sw;
      return sub_buffer[cnt];
    }

    //sub_A = sub(A, box)
    template <typename T>
    void get_buffer_subarray(T *sub_buffer, T *buffer, const Box &sub_box,
      const Box &box, int sw) {
      
      Shape sp = box.shape_with_stencil(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
			
      Box bd_box = box.boundary_box(sw);
      Box ref_box = sub_box.ref_box(bd_box);
      int xs, xe, ys, ye, zs, ze;
      ref_box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
      
      //ref_box.display("ref_box");

      int cnt = 0;
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
#pragma simd
          for (int i = xs; i < xe; i++) {
            sub_buffer[(k-zs)*(ye-ys)*(xe-xs)+(j-ys)*(xe-xs)+(i-xs)] = buffer[k * M * N + j * M + i];
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    // set sub(A) = B
    template<typename T1, typename T2>
    void copy_buffer(
        T1* A_buf,
        const Shape& A_buf_shape,
        const Box&  A_window,
        T2* B_buf,
        const Shape& B_buf_shape,
        const Box& B_window) {

      Shape sp = A_window.shape();
      const int M = sp[0];
      const int N = sp[1];
      const int P = sp[2];
      
      const int M1 = A_buf_shape[0];
      const int N1 = A_buf_shape[1];

      const int M2 = B_buf_shape[0];
      const int N2 = B_buf_shape[1];      

      int xs1, xe1, ys1, ye1, zs1, ze1;
      A_window.get_corners(xs1, xe1, ys1, ye1, zs1, ze1);

      int xs2, xe2, ys2, ye2, zs2, ze2;
      B_window.get_corners(xs2, xe2, ys2, ye2, zs2, ze2);
      
      //ref_box.display("ref_box");

      int cnt = 0;
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
#pragma simd
          for (int i = 0; i < M; i++) {
            A_buf[(k+zs1)*M1*N1 + (j+ys1)*M1 + i + xs1] =
              B_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i+xs2];
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    template<typename T1, typename T2, typename T3>
    void copy_buffer_with_mask(
        T1* A_buf,
        const Shape& A_buf_shape,
        const Box&  A_window,
        T2* B_buf,
        const Shape& B_buf_shape,
        const Box& B_window, 
        T3* C_buf,
        const Shape& C_buf_shape,
        const Box& C_window,
        bool ifscalar_B
        ) {

      Shape sp = A_window.shape();
      const int M = sp[0];
      const int N = sp[1];
      const int P = sp[2];
      
      const int M1 = A_buf_shape[0];
      const int N1 = A_buf_shape[1];

      const int M2 = C_buf_shape[0];
      const int N2 = C_buf_shape[1];      

      int xs1, xe1, ys1, ye1, zs1, ze1;
      A_window.get_corners(xs1, xe1, ys1, ye1, zs1, ze1);

      int xs2, xe2, ys2, ye2, zs2, ze2;
      C_window.get_corners(xs2, xe2, ys2, ye2, zs2, ze2);

      //ref_box.display("ref_box");
      if(!ifscalar_B){
        int cnt = 0;
        for (int k = 0; k < P; k++) {
          for (int j = 0; j < N; j++) {
#pragma simd
            for (int i = 0; i < M; i++) {
              if(C_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i+xs2] > 0)// >0
                A_buf[(k+zs1)*M1*N1 + (j+ys1)*M1 + i + xs1] =
                  B_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i+xs2];
            }
          }
        }
      }
      else{
        int cnt = 0;
        for (int k = 0; k < P; k++) {
          for (int j = 0; j < N; j++) {
#pragma simd
            for (int i = 0; i < M; i++) {
              if(C_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i+xs2] > 0)// >0
                A_buf[(k+zs1)*M1*N1 + (j+ys1)*M1 + i + xs1] = B_buf[0];
            }
          }
        }
      }

    }


    // set sub(A) = const
    template<typename T1, typename T2>
    void set_buffer_subarray_const(T1* buffer, T2 val, const Box &box, 
      const Box &sub_box, int sw) {

      Shape sp = box.shape_with_stencil(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
      
      Box bd_box = box.boundary_box(sw);
      Box ref_box = sub_box.ref_box(bd_box);
      int xs, xe, ys, ye, zs, ze;
      ref_box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
      
      //ref_box.display("ref_box");

      for (int k = zs+sw; k < ze-sw; k++) {
        for (int j = ys+sw; j < ye-sw; j++) {
#pragma simd
          for (int i = xs+sw; i < xe-sw; i++) {
            buffer[k * M * N + j * M + i] = val;
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }



    template<typename T>
    void set_buffer_rand(T *buffer, int size) {
      arma::Col<T> v = oa::utils::make_vec(size, buffer);
      arma::arma_rng::set_seed_random();
      v.randu();
    }

    //specialized for int type
    template<>
    void set_buffer_rand(int *buffer, int size);

    
    template<typename T>
    void set_buffer_seqs(T *buffer, const Shape& s, Box box, int sw) {
      int cnt = 0;
      int xs, xe, ys, ye, zs, ze;
      int M = s[0];
      int N = s[1];
      int P = s[2];
      //cout<<M<<" "<<N<<" "<<P<<endl;
      box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
      //printf("%d %d %d %d %d %d\n", xs, xe, ys, ye, zs, ze);
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
#pragma simd
          for (int i = xs; i < xe; i++) {
            buffer[(k-zs)*(ye-ys)*(xe-xs)+(j-ys)*(xe-xs)+(i-xs)] = k * M * N + j * M + i;
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    template <typename T>
    void set_ghost_consts(T *buffer, const Shape &sp, T val, int sw = 1) {
      int M = sp[0] + 2 * sw;
      int N = sp[1] + 2 * sw;
      int P = sp[2] + 2 * sw;

      int cnt = 0;
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
#pragma simd
          for (int i = 0; i < M; i++) {
            if ((sw <= k && k < P - sw) &&
                (sw <= j && j < N - sw) &&
                (sw <= i && i < M - sw)) {
              cnt++;
              continue;
            }
            buffer[cnt++] = val;
          }
        }
      }
    }

    int3 calc_step(Box box, int d, int sw);

    // A = B
    template<typename T>
    void copy_buffer(T *A, T *B, int size) {
#pragma simd
      for (int i = 0; i < size; i++) {
        A[i] = B[i];
      }
    }





    // template<typename T>
    // inline int calc_id(int i, int j, int k, int3 S) {
    //   int M = S[0];
    //   int N = S[1];
    //   int P = S[2];
    //   return k * M * N + j * M + i;
    // }



  }
}

#endif
