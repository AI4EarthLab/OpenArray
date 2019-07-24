
#include <iostream>

namespace oa{
  namespace internal{

    template<typename T>
    void buffer_max_const(T &val, int* pos, T *A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      // printf("(%d-%d,%d-%d,%d-%d)\n", xs, xe, ys, ye, zs, ze);
      // printf("(%d,%d,%d)\n", pos[0], pos[1], pos[2]);
      
      int M = as[0];
      int N = as[1];
      int K = as[2];

      
      val = (A[zs * M * N + ys * M + xs]);

      pos[0] = xs;
      pos[1] = ys;
      pos[2] = zs;

      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if ((A[i + j * M + k * M * N]) > val) {
              val = (A[i + j * M + k * M * N]);
              pos[0] = i; 
              pos[1] = j;
              pos[2] = k;
            }
          }
        }
      }

    }
    template<typename T>
    void buffer_min_const(T &val, int* pos, T *A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      // printf("(%d-%d,%d-%d,%d-%d)\n", xs, xe, ys, ye, zs, ze);
      // printf("(%d,%d,%d)\n", pos[0], pos[1], pos[2]);
      
      int M = as[0];
      int N = as[1];
      int K = as[2];

      
      val = (A[zs * M * N + ys * M + xs]);

      pos[0] = xs;
      pos[1] = ys;
      pos[2] = zs;

      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if ((A[i + j * M + k * M * N]) < val) {
              val = (A[i + j * M + k * M * N]);
              pos[0] = i; 
              pos[1] = j;
              pos[2] = k;
            }
          }
        }
      }

    }
    template<typename T>
    void buffer_abs_max_const(T &val, int* pos, T *A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      // printf("(%d-%d,%d-%d,%d-%d)\n", xs, xe, ys, ye, zs, ze);
      // printf("(%d,%d,%d)\n", pos[0], pos[1], pos[2]);
      
      int M = as[0];
      int N = as[1];
      int K = as[2];

      
      val = std::abs(A[zs * M * N + ys * M + xs]);

      pos[0] = xs;
      pos[1] = ys;
      pos[2] = zs;

      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if (std::abs(A[i + j * M + k * M * N]) > val) {
              val = std::abs(A[i + j * M + k * M * N]);
              pos[0] = i; 
              pos[1] = j;
              pos[2] = k;
            }
          }
        }
      }

    }
    template<typename T>
    void buffer_abs_min_const(T &val, int* pos, T *A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      // printf("(%d-%d,%d-%d,%d-%d)\n", xs, xe, ys, ye, zs, ze);
      // printf("(%d,%d,%d)\n", pos[0], pos[1], pos[2]);
      
      int M = as[0];
      int N = as[1];
      int K = as[2];

      
      val = std::abs(A[zs * M * N + ys * M + xs]);

      pos[0] = xs;
      pos[1] = ys;
      pos[2] = zs;

      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if (std::abs(A[i + j * M + k * M * N]) < val) {
              val = std::abs(A[i + j * M + k * M * N]);
              pos[0] = i; 
              pos[1] = j;
              pos[2] = k;
            }
          }
        }
      }

    }



    template<typename TC, typename TA, typename TB>
    void buffer_max2(
        TC* C, TA *A, TB *B,
        const Shape& sc,
        const Shape& sa,
        const Shape& sb,
        const Box& wc,
        const Box& wa,
        const Box& wb) {
      
      int x = 0, y = 0, z = 0;
      int xs_a, xe_a, ys_a, ye_a, zs_a, ze_a;
      wa.get_corners(xs_a, xe_a, ys_a, ye_a, zs_a, ze_a);

      int xs_b, xe_b, ys_b, ye_b, zs_b, ze_b;
      wb.get_corners(xs_b, xe_b, ys_b, ye_b, zs_b, ze_b);

      int xs_c, xe_c, ys_c, ye_c, zs_c, ze_c;
      wc.get_corners(xs_c, xe_c, ys_c, ye_c, zs_c, ze_c);
      
      int M = xe_a - xs_a;
      int N = ye_a - ys_a;
      int P = ze_a - zs_a;

      const int MC = sc[0];
      const int NC = sc[1];
      const int PC = sc[2];

      const int MB = sb[0];
      const int NB = sb[1];
      const int PB = sb[2];

      const int MA = sa[0];
      const int NA = sa[1];
      const int PA = sa[2];

      for(int k = 0; k < P; ++k){
        for(int j = 0; j < N; ++j){
          for(int i = 0; i < M; ++i){
            const int ia = (k + zs_a) * MA * NA +
              (j + ys_a) * MA + (i + xs_a);
            const int ib = (k + zs_b) * MB * NB +
              (j + ys_b) * MB + (i + xs_b);
            const int ic = (k + zs_c) * MC * NC +
              (j + ys_c) * MC + (i + xs_c);
            C[ic] = (A[ia] > B[ib]) ? A[ia] : B[ib];
          }
        }
      }
      //std::cout<<"MNK:"<<M<<" "<<N<<" "<<" "
      //<<K<<" sw="<<sw<<std::endl;
    }
    template<typename TC, typename TA, typename TB>
    void buffer_min2(
        TC* C, TA *A, TB *B,
        const Shape& sc,
        const Shape& sa,
        const Shape& sb,
        const Box& wc,
        const Box& wa,
        const Box& wb) {
      
      int x = 0, y = 0, z = 0;
      int xs_a, xe_a, ys_a, ye_a, zs_a, ze_a;
      wa.get_corners(xs_a, xe_a, ys_a, ye_a, zs_a, ze_a);

      int xs_b, xe_b, ys_b, ye_b, zs_b, ze_b;
      wb.get_corners(xs_b, xe_b, ys_b, ye_b, zs_b, ze_b);

      int xs_c, xe_c, ys_c, ye_c, zs_c, ze_c;
      wc.get_corners(xs_c, xe_c, ys_c, ye_c, zs_c, ze_c);
      
      int M = xe_a - xs_a;
      int N = ye_a - ys_a;
      int P = ze_a - zs_a;

      const int MC = sc[0];
      const int NC = sc[1];
      const int PC = sc[2];

      const int MB = sb[0];
      const int NB = sb[1];
      const int PB = sb[2];

      const int MA = sa[0];
      const int NA = sa[1];
      const int PA = sa[2];

      for(int k = 0; k < P; ++k){
        for(int j = 0; j < N; ++j){
          for(int i = 0; i < M; ++i){
            const int ia = (k + zs_a) * MA * NA +
              (j + ys_a) * MA + (i + xs_a);
            const int ib = (k + zs_b) * MB * NB +
              (j + ys_b) * MB + (i + xs_b);
            const int ic = (k + zs_c) * MC * NC +
              (j + ys_c) * MC + (i + xs_c);
            C[ic] = (A[ia] < B[ib]) ? A[ia] : B[ib];
          }
        }
      }
      //std::cout<<"MNK:"<<M<<" "<<N<<" "<<" "
      //<<K<<" sw="<<sw<<std::endl;
    }
  }
}
