
#ifndef __SUM_INTERNAL_GPU_HPP__
#define __SUM_INTERNAL_GPU_HPP__

#ifdef __HAVE_CUDA__

namespace oa {
namespace internal {
namespace gpu {

    template <typename T>
    extern void ker_buffer_sum_scalar_const(T *val, T *A, const int* host_values, const int sw, const int size);
    template <typename T>
    void buffer_sum_scalar_const(T *val, T *A, Box box, int sw, int size) {
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
        
        int *host_values = (int*)malloc(sizeof(int)*6);
        host_values[0] = xs; host_values[1] = xe;
        host_values[2] = ys; host_values[3] = ye;
        host_values[4] = zs; host_values[5] = ze;

        oa::internal::gpu::ker_buffer_sum_scalar_const(val, A, host_values, sw, size);

        free(host_values);
    }

    template<typename T>
    extern void ker_buffer_csum_x_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type);
    template<typename T>
    void buffer_csum_x_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
        
        int *host_values = (int*)malloc(sizeof(int)*6);
        host_values[0] = xs; host_values[1] = xe;
        host_values[2] = ys; host_values[3] = ye;
        host_values[4] = zs; host_values[5] = ze;

        oa::internal::gpu::ker_buffer_csum_x_const(ap, A, host_values, sw, size, buffer, type);

        free(host_values);
    }

    template<typename T>
    extern void ker_buffer_csum_y_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type);
    template<typename T>
    void buffer_csum_y_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
        
        int *host_values = (int*)malloc(sizeof(int)*6);
        host_values[0] = xs; host_values[1] = xe;
        host_values[2] = ys; host_values[3] = ye;
        host_values[4] = zs; host_values[5] = ze;

        oa::internal::gpu::ker_buffer_csum_y_const(ap, A, host_values, sw, size, buffer, type);

        free(host_values);
    }

    template<typename T>
    extern void ker_buffer_csum_z_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type);
    template<typename T>
    void buffer_csum_z_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
        
        int *host_values = (int*)malloc(sizeof(int)*6);
        host_values[0] = xs; host_values[1] = xe;
        host_values[2] = ys; host_values[3] = ye;
        host_values[4] = zs; host_values[5] = ze;

        oa::internal::gpu::ker_buffer_csum_z_const(ap, A, host_values, sw, size, buffer, type);

        free(host_values);
    }

}}}
#endif

#endif
