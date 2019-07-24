#ifndef __MINMAX_INTERNAL_GPU_HPP__
#define __MINMAX_INTERNAL_GPU_HPP__

#ifdef __HAVE_CUDA__

namespace oa {
namespace internal {
namespace gpu {
    template <typename T>
    extern void ker_buffer_max_const(T *A, int* host_values, T &val, int *pos);
    template <typename T>
    extern void ker_buffer_min_const(T *A, int* host_values, T &val, int *pos);
    template <typename T>
    extern void ker_buffer_abs_max_const(T *A, int* host_values, T &val, int *pos);
    template <typename T>
    extern void ker_buffer_abs_min_const(T *A, int* host_values, T &val, int *pos);


    template <typename T>
    void buffer_max_const(T &host_val, int* host_pos, T *dev_A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      int* host_values = (int*)malloc(sizeof(int)*9);
      host_values[0] = as[0]; host_values[1] = as[1]; host_values[2] = as[2];
      host_values[3] = xs; host_values[4] = xe;
      host_values[5] = ys; host_values[6] = ye;
      host_values[7] = zs; host_values[8] = ze;

      oa::internal::gpu::ker_buffer_max_const(dev_A, host_values, host_val, host_pos);
      free(host_values);

    }

    template <typename T>
    void buffer_min_const(T &host_val, int* host_pos, T *dev_A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      int* host_values = (int*)malloc(sizeof(int)*9);
      host_values[0] = as[0]; host_values[1] = as[1]; host_values[2] = as[2];
      host_values[3] = xs; host_values[4] = xe;
      host_values[5] = ys; host_values[6] = ye;
      host_values[7] = zs; host_values[8] = ze;

      oa::internal::gpu::ker_buffer_min_const(dev_A, host_values, host_val, host_pos);
      free(host_values);

    }

    template <typename T>
    void buffer_abs_max_const(T &host_val, int* host_pos, T *dev_A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      int* host_values = (int*)malloc(sizeof(int)*9);
      host_values[0] = as[0]; host_values[1] = as[1]; host_values[2] = as[2];
      host_values[3] = xs; host_values[4] = xe;
      host_values[5] = ys; host_values[6] = ye;
      host_values[7] = zs; host_values[8] = ze;

      oa::internal::gpu::ker_buffer_abs_max_const(dev_A, host_values, host_val, host_pos);
      free(host_values);

    }

    template <typename T>
    void buffer_abs_min_const(T &host_val, int* host_pos, T *dev_A,
            const Shape& as,
            Box win) {

      int xs, xe, ys, ye, zs, ze;
      win.get_corners(xs, xe, ys, ye, zs, ze);

      int* host_values = (int*)malloc(sizeof(int)*9);
      host_values[0] = as[0]; host_values[1] = as[1]; host_values[2] = as[2];
      host_values[3] = xs; host_values[4] = xe;
      host_values[5] = ys; host_values[6] = ye;
      host_values[7] = zs; host_values[8] = ze;

      oa::internal::gpu::ker_buffer_abs_min_const(dev_A, host_values, host_val, host_pos);
      free(host_values);

    }


    template<typename TC, typename TA, typename TB>
    extern void ker_buffer_max2(TC* C, TA *A, TB *B, int* host_values);
    template<typename TC, typename TA, typename TB>
    extern void ker_buffer_min2(TC* C, TA *A, TB *B, int* host_values);


    template<typename TC, typename TA, typename TB>
    void buffer_max2(
        TC* dev_C, TA *dev_A, TB *dev_B,
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

      int* host_values = (int*)malloc(sizeof(int)*27);
      host_values[0] = xs_a;
      host_values[1] = xe_a;
      host_values[2] = ys_a;
      host_values[3] = ye_a;
      host_values[4] = zs_a;
      host_values[5] = ze_a;
      host_values[6] = xs_b;
      host_values[7] = xe_b;
      host_values[8] = ys_b;
      host_values[9] = ye_b;
      host_values[10] = zs_b;
      host_values[11] = ze_b;
      host_values[12] = xs_c;
      host_values[13] = xe_c;
      host_values[14] = ys_c;
      host_values[15] = ye_c;
      host_values[16] = zs_c;
      host_values[17] = ze_c;

      host_values[18] = MA;
      host_values[19] = NA;
      host_values[20] = PA;
      host_values[21] = MB;
      host_values[22] = NB;
      host_values[23] = PB;
      host_values[24] = MC;
      host_values[25] = NC;
      host_values[26] = PC;
    
      oa::internal::gpu::ker_buffer_max2(dev_C, dev_A, dev_B, host_values);

      free(host_values);
    }

    template<typename TC, typename TA, typename TB>
    void buffer_min2(
        TC* dev_C, TA *dev_A, TB *dev_B,
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

      int* host_values = (int*)malloc(sizeof(int)*27);
      host_values[0] = xs_a;
      host_values[1] = xe_a;
      host_values[2] = ys_a;
      host_values[3] = ye_a;
      host_values[4] = zs_a;
      host_values[5] = ze_a;
      host_values[6] = xs_b;
      host_values[7] = xe_b;
      host_values[8] = ys_b;
      host_values[9] = ye_b;
      host_values[10] = zs_b;
      host_values[11] = ze_b;
      host_values[12] = xs_c;
      host_values[13] = xe_c;
      host_values[14] = ys_c;
      host_values[15] = ye_c;
      host_values[16] = zs_c;
      host_values[17] = ze_c;

      host_values[18] = MA;
      host_values[19] = NA;
      host_values[20] = PA;
      host_values[21] = MB;
      host_values[22] = NB;
      host_values[23] = PB;
      host_values[24] = MC;
      host_values[25] = NC;
      host_values[26] = PC;
    
      oa::internal::gpu::ker_buffer_min2(dev_C, dev_A, dev_B, host_values);

      free(host_values);
    }
}}}

#endif

#endif
