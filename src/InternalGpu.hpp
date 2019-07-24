#ifndef __INTERNAL_GPU_HPP__
#define __INTERNAL_GPU_HPP__

#ifdef __HAVE_CUDA__
#include <random>
#include "common.hpp"
#include "Box.hpp"
#include "utils/utils.hpp"
#include <vector>
#include "CUDA.hpp"
#include "Array.hpp"
#include "Partition.hpp"

using namespace std;

namespace oa {
  namespace gpu {

    template <typename T>
    extern void set_buffer_consts(T *buffer, int n, T val);

    template <typename T>
    extern void set_buffer_local(T* sub_buffer, const Box& box, int x, int y, int z, T val, int sw);

    template <typename T>
    extern void get_buffer_subarray(T *sub_buffer, T *buffer, const Box &sub_box, const Box &box, int sw);

    template<typename T1, typename T2>
    extern void copy_buffer(
        T1* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
        T2* B_buf, const Shape& B_buf_shape, const Box& B_window); 
    
    template<typename T>
    extern void copy_buffer(T *A, T *B, int size);

    template<typename T1, typename T2, typename T3>
    extern void copy_buffer_with_mask( T1* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
        T2* B_buf, const Shape& B_buf_shape, const Box& B_window, 
        T3* C_buf, const Shape& C_buf_shape, const Box& C_window,bool ifscalar_B);

    template<typename T>
    extern void set_buffer_seqs(T *buffer, const Shape& s, Box box, int sw);

    template<typename T>
    extern ArrayPtr set_local_array(const Shape& gs, T* buf);

    template<typename T>
    extern void set_ghost_consts(T *buffer, const Shape &sp, T val, int sw = 1);

    template<typename T1, typename T2>
    extern void set_buffer_subarray_const(T1* buffer, T2 val, const Box &box, 
       const Box &sub_box, int sw);

    template<typename T>
    extern bool is_equal_arrayptr_and_array(const ArrayPtr& A, T* B);

    template<typename T>
    extern bool is_equal_arrayptr_and_scalar(const ArrayPtr& A, T B);

    template<typename T>
    extern bool is_equal_array_and_array(T* buf_A, T* buf_B, int n);

    template <typename T>
    T get_buffer_local_sub(T* sub_buffer, const Box& box, int x, int y, int z, int sw) {
       Shape sp = box.shape_with_stencil(sw);
       int M = sp[0];
       int N = sp[1];
       int P = sp[2];
       
       int cnt = (z + sw) * M * N + (y + sw) * M + x + sw;
       T h_res;
       CUDA_CHECK(cudaMemcpy(&h_res, sub_buffer+cnt, sizeof(T), cudaMemcpyDeviceToHost));
       return h_res;
    }
  }
}

#endif
#endif
