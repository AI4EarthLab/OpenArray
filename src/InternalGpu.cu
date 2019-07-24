#ifndef __INTERNEL_GPU_CPP__
#define __INTERNEL_GPU_HPP__
#ifdef __HAVE_CUDA__
#include "InternalGpu.hpp"
#include <random>
#include "common.hpp"
#include "Box.hpp"
#include "utils/utils.hpp"
#include <vector>
#include "CUDA.hpp"
#include "Function.hpp"

namespace oa {
  namespace gpu {
  
  //GPU kernels
    template <typename T>
    __global__ void ker_set_buffer_consts(T *buffer, int n, T val) {
      int id = blockIdx.x*blockDim.x+threadIdx.x;
      if (id < n)
         buffer[id] = val;
       //__syncthreads();
    }
   
   template <typename T>
  __global__ void ker_set_buffer_local(T *buffer, int cnt, T val){
	    buffer[cnt] = val;
       //__syncthreads();
    }
    
    template <typename T>
    __global__ void ker_get_buffer_subarray(T *sub_buffer, T *buffer, 
      int xs, int xe, int ys, int ye, int zs, int ze, int M, int N) {
        int i = blockIdx.x * blockDim.x + threadIdx.x + xs;
        int j = blockIdx.y * blockDim.y + threadIdx.y + ys;
        int k = blockIdx.z + zs;
        if(i >= xe || j >= ye || k>=ze ) return;
        sub_buffer[(k-zs)*(ye-ys)*(xe-xs)+(j-ys)*(xe-xs)+(i-xs)] = buffer[k * M * N + j * M + i];
        //__syncthreads();
    }

    template <typename T1, typename T2>
    __global__  void ker_copy_buffer(T1* A_buf, T2* B_buf, int xs1, int xe1, int ys1, int ye1, int zs1, int ze1,
                     int xs2, int xe2, int ys2, int ye2, int zs2, int ze2,
                     const int M, const int N, const int P, const int M1, const int N1, const int M2, const int N2){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z;
        if(i>=M || j>=N || k>=P) return;
        A_buf[(k+zs1)*M1*N1 + (j+ys1)*M1 + i + xs1] =
        B_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i + xs2];
        //__syncthreads();
      }
    
    template <typename T> 
    __global__ void ker_copy_buffer(T* A, T* B, int n){
      int id = blockIdx.x*blockDim.x+threadIdx.x;
      if (id < n)
         A[id] = B[id];
       //__syncthreads();

    }

     template <typename T1, typename T2, typename T3>
     __global__  void ker_copy_buffer_with_mask(T1* A_buf, T2* B_buf, T3* C_buf, int xs1, int xe1, int ys1, int ye1, int zs1, int ze1,
                      int xs2, int xe2, int ys2, int ye2, int zs2, int ze2,
                      const int M, const int N, const int P,  const int M1, const int N1, const int M2, const int N2, bool ifscalar_B){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z;
        int B_buf_id = (ifscalar_B == false ? (k+zs2)*M2*N2 + (j+ys2)*M2 + i + xs2 : 0);
        if(i>=M || j>=N || k>=P) return;
        if(C_buf[(k+zs2)*M2*N2 + (j+ys2)*M2 + i + xs2] > 0){
           A_buf[(k+zs1)*M1*N1 + (j+ys1)*M1 + i + xs1] = B_buf[B_buf_id];
        }
        //__syncthreads();
      }

      template<typename T>
      __global__ void ker_set_buffer_seqs(T *buffer, int xs, int xe, int ys, int ye, int zs, int ze, int M, int N){
        int i = blockIdx.x * blockDim.x + threadIdx.x + xs;
        int j = blockIdx.y * blockDim.y + threadIdx.y + ys;
        int k = blockIdx.z + zs;
        if(i>=xe || j>=ye  || k>=ze) return;
        buffer[(k-zs)*(ye-ys)*(xe-xs)+(j-ys)*(xe-xs)+(i-xs)] = (T)k * M * N + j * M + i;
        //__syncthreads();
      }

      template <typename T>
      __global__ void ker_set_ghost_consts(T *buffer, int M, int N, int P, int sw, T val){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int k = blockIdx.z;
        if ((sw <= k && k < P - sw) &&
        (sw <= j && j < N - sw) &&
        (sw <= i && i < M - sw)) return ;
        buffer[k*M*N + j*M + i] = val;
        //__syncthreads();
      }

      template<typename T> 
      __global__ void ker_set_local_array(T* A_buf, T* B_buf, int xs, int xe, int ys, int ye, int zs, int ze, int M, int N, int sw, int gs0, int gs1){
        int i = blockIdx.x * blockDim.x + threadIdx.x + xs + sw;
        int j = blockIdx.y * blockDim.y + threadIdx.y + ys + sw;
        int k = blockIdx.z + zs + sw;
        if(i>=xe-sw || j>=ye-sw || k>=ze-sw) return;
        A_buf[i+j*M+k*M*N] = B_buf[i-sw + (j-sw) * gs0 + (k-sw)*gs0*gs1];
        //__syncthreads();
      }

      template<typename T1, typename T2>
     __global__ void ker_set_buffer_subarray_const(T1* buffer, T2 val, int xs, int xe, int ys, int ye, int zs, int ze,  int M, int N, int sw){
        int i = blockIdx.x * blockDim.x + threadIdx.x + xs + sw;
        int j = blockIdx.y * blockDim.y + threadIdx.y + ys + sw;
        int k = blockIdx.z + zs +sw;
        if(i>=xe-sw || j>=ye-sw || k>=ze-sw) return;
        buffer[k * M * N + j * M + i] = val;
        //__syncthreads();
      }
 
    //Functions for other files  
    template <typename T>
    void set_buffer_consts(T *buffer, int n, T val){
      auto tb = SizeToBlockThreadPair(n);
      int total_size = tb.first*tb.second;
      for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
       ker_set_buffer_consts<<<tb.first, tb.second>>>(buffer+curr_pos, min(total_size, n-curr_pos), val);
    }



   template <typename T>
   void set_buffer_local(T* sub_buffer, const Box& box, int x, int y, int z, T val, int sw) {
      Shape sp = box.shape_with_stencil(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
      int cnt = (z + sw) * M * N + (y + sw) * M + x + sw;
      ker_set_buffer_local<<<1,1>>>(sub_buffer, cnt, val);
    }

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
      bool layer = false;
      if(xs == 0 && xe == M && ys == 0 && ye == N)
        layer = true;
      if(!layer){
        dim3 threads_per_block(16,16);
        dim3 num_blocks((xe-xs+15)/16,(ye-ys+15)/16, ze);
        ker_get_buffer_subarray<<<num_blocks, threads_per_block>>>(sub_buffer, buffer,xs, xe, ys, ye, zs, ze, M, N); 
      }else{
        //CUDA_CHECK(cudaMemcpy(sub_buffer, buffer+zs*M*N, (ze-zs)*M*N*sizeof(T), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(sub_buffer+sw*M*N, buffer+zs*M*N+sw*M*N, (ze-zs-sw)*M*N*sizeof(T), cudaMemcpyDeviceToDevice));
      }

    }

    template<typename T1, typename T2>
    void copy_buffer(
      T1* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
      T2* B_buf, const Shape& B_buf_shape, const Box& B_window){
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
      dim3 threads_per_block(16,16);
      dim3 num_blocks((M+15)/16,(N+15)/16, P);
      ker_copy_buffer<<<num_blocks, threads_per_block>>>(A_buf, B_buf, xs1, xe1, ys1, ye1, zs1, ze1,
                                                   xs2, xe2, ys2, ye2, zs2, ze2,
                                                   M, N, P,  M1, N1, M2, N2);
    }

    template<typename T>
    void copy_buffer(T *A, T *B, int n) {
      auto tb = SizeToBlockThreadPair(n);
      int total_size = tb.first*tb.second;
      for(int curr_pos = 0; curr_pos < n; curr_pos += total_size)
        ker_copy_buffer<<<tb.first, tb.second>>>(A+curr_pos, B+curr_pos, min(total_size, n-curr_pos));
    }

    template<typename T1, typename T2, typename T3>
    void copy_buffer_with_mask( T1* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
          T2* B_buf, const Shape& B_buf_shape, const Box& B_window, 
          T3* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B){
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
      dim3 threads_per_block(16,16);
      dim3 num_blocks((M+15)/16,(N+15)/16, P);
      ker_copy_buffer_with_mask<<<num_blocks, threads_per_block>>>(A_buf, B_buf, C_buf, xs1, xe1, ys1, ye1, zs1, ze1,
                  xs2, xe2, ys2, ye2, zs2, ze2,
                  M, N, P, M1, N1, M2, N2, ifscalar_B);

    }
    
    template<typename T>
    void set_buffer_seqs(T *buffer, const Shape& s, Box box, int sw) {
      int cnt = 0;
      int xs, xe, ys, ye, zs, ze;
      int M = s[0];
      int N = s[1];
      int P = s[2];
      box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
      dim3 threads_per_block(16,16);
      dim3 num_blocks((xe-xs+15)/16,(ye-ys+15)/16, ze-zs);
      ker_set_buffer_seqs<<<num_blocks, threads_per_block>>>(buffer, xs, xe, ys, ye, zs, ze, M, N);
    }

    template <typename T>
    void set_ghost_consts(T *buffer, const Shape &sp, T val, int sw) {
      int M = sp[0] + 2 * sw;
      int N = sp[1] + 2 * sw;
      int P = sp[2] + 2 * sw;

      int cnt = 0;
      dim3 threads_per_block(16,16);
      dim3 num_blocks((M+15)/16,(N+15)/16, P);
        ker_set_ghost_consts<<<num_blocks, threads_per_block>>>(buffer, M, N, P, sw, val);
    }
    
    template <typename T> 
    ArrayPtr set_local_array(const Shape& gs, T* buf){
      int sw = Partition::get_default_stencil_width();
      DataType dt = oa::utils::to_type<T>();        
      ArrayPtr ap = oa::funcs::zeros(MPI_COMM_SELF, gs, sw, dt);
      T* dst_buf = (T*)ap->get_buffer();

      const int xs = 0;
      const int xe = gs[0] + 2 * sw;
      const int ys = 0;
      const int ye = gs[1] + 2 * sw;
      const int zs = 0;
      const int ze = gs[2] + 2 * sw;

      const int M = xe;
      const int N = ye;
      const int P = ze;
      
      dim3 threads_per_block(16,16);
      dim3 num_blocks((xe-xs+15)/16,(ye-ys+15)/16, ze-zs);
      ker_set_local_array<<<num_blocks, threads_per_block>>>(dst_buf, buf, xs, xe, ys, ye, zs, ze,  M, N,  sw, gs[0],gs[1]);
      return ap;
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
       dim3 threads_per_block(16,16);
       dim3 num_blocks((xe-xs+15)/16,(ye-ys+15)/16, ze-zs);
       ker_set_buffer_subarray_const<<<num_blocks, threads_per_block>>>(buffer, val, xs, xe, ys, ye, zs,ze, M, N, sw);
     }
 
     template<typename T1, typename T2>
     __global__ void ker_is_equal_arrayptr_and_array(T1* A_buf, T2* B_buf, int* d_is_equal, int s0, int s1, int k, int sw){
      int i = blockIdx.x * blockDim.x + threadIdx.x + sw;
      int j = blockIdx.y * blockDim.y + threadIdx.y + sw;
      if(i>=s0-sw || j>=s1-sw) return;
      int id_a = i + j * s0 + k * s0 * s1;
      int id_b = (i-sw) + (j-sw)*s0 + (k-sw)*s0*s1;
      if(*d_is_equal == 0) return;
      int d_is_equal_element = 1;
      if(A_buf[id_a]-B_buf[id_b]>1E-6 || A_buf[id_a]-B_buf[id_b]<-1E-6) d_is_equal_element = 0;
      atomicAnd(d_is_equal, d_is_equal_element);
     }
     
     template<typename T>
     bool is_equal_arrayptr_and_array(const ArrayPtr& A, T* B){
       if (!A->is_seqs()) return false;
 
       if(A->get_data_type() == DATA_INT){
         int* A_buf = (int*)A->get_buffer();
         Shape s = A->buffer_shape();
         const int sw = A->get_partition()->get_stencil_width();
 
         int cnt = 0;
         int* d_is_equal;
         int h_is_equal = 1;
         CUDA_CHECK(cudaMalloc((void**)&d_is_equal, sizeof(int)));
         CUDA_CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(int), cudaMemcpyHostToDevice));
         dim3 threads_per_block(16,16);
         dim3 num_blocks((s[0]+15)/16,(s[1]+15)/16);
         for(int k = sw; k < s[2] - sw; k++){
           ker_is_equal_arrayptr_and_array<<<num_blocks, threads_per_block>>>(A_buf, B, d_is_equal, s[0], s[1], k, sw);
         }
         CUDA_CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(int), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaFree(d_is_equal));
         return h_is_equal==1;
       }
       if(A->get_data_type() == DATA_FLOAT){
         float* A_buf = (float*)A->get_buffer();
         Shape s = A->buffer_shape();
         const int sw = A->get_partition()->get_stencil_width();
 
         int cnt = 0;
         int* d_is_equal;
         int h_is_equal = 1;
         CUDA_CHECK(cudaMalloc((void**)&d_is_equal, sizeof(int)));
         CUDA_CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(int), cudaMemcpyHostToDevice));
         dim3 threads_per_block(16,16);
         dim3 num_blocks((s[0]+15)/16,(s[1]+15)/16);
         for(int k = sw; k < s[2] - sw; k++){
           ker_is_equal_arrayptr_and_array<<<num_blocks, threads_per_block>>>(A_buf, B, d_is_equal, s[0], s[1], k, sw);
         }
         CUDA_CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(int), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaFree(d_is_equal));
         return h_is_equal==1;
       }
       if(A->get_data_type() == DATA_DOUBLE){
         double* A_buf = (double*)A->get_buffer();
         Shape s = A->buffer_shape();
         const int sw = A->get_partition()->get_stencil_width();
 
         int cnt = 0;
         int* d_is_equal;
         int h_is_equal = 1;
         CUDA_CHECK(cudaMalloc((void**)&d_is_equal, sizeof(int)));
         CUDA_CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(int), cudaMemcpyHostToDevice));
         dim3 threads_per_block(16,16);
         dim3 num_blocks((s[0]+15)/16,(s[1]+15)/16);
         for(int k = sw; k < s[2] - sw; k++){
           ker_is_equal_arrayptr_and_array<<<num_blocks, threads_per_block>>>(A_buf, B, d_is_equal, s[0], s[1], k, sw);
         }
         CUDA_CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(int), cudaMemcpyDeviceToHost));
         CUDA_CHECK(cudaFree(d_is_equal));
         return h_is_equal==1;
       }
       return false;
     }
     
     template<typename T>
     __global__ void ker_is_equal_arrayptr_and_scalar(T* A, T B, int* d_is_equal){
       if(A[0]-B>1E-6 || A[0]-B<-1E-6) *d_is_equal = 0;
       else *d_is_equal = 1;
     }

     template<typename T>
     bool is_equal_arrayptr_and_scalar(const ArrayPtr& A, T B){
      if (!A->is_seqs_scalar()) return false;
      int* d_is_equal;
      int h_is_equal = 1;

      CUDA_CHECK(cudaMalloc((void**)&d_is_equal, sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(int), cudaMemcpyHostToDevice));
      T* A_buf = (T*)A->get_buffer();
      ker_is_equal_arrayptr_and_scalar<<<1,1>>>(A_buf, B, d_is_equal);
      CUDA_CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_is_equal));
      return h_is_equal==1;
     }

    template<typename T>
    __global__ void ker_is_equal_array_and_array(T* buf_A, T* buf_B, int n, int* d_is_equal){
      int id = blockIdx.x*blockDim.x+threadIdx.x;
      if (id >= n || *d_is_equal == 0) return;
      int d_is_equal_element = 1;
      if( buf_A[id] - buf_B[id] > 1E-8 || buf_A[id] - buf_B[id]<-1E-8)
        d_is_equal_element = 0;
      atomicAnd(d_is_equal, d_is_equal_element);
       //__syncthreads();
    }

    template<typename T>
     bool is_equal_array_and_array(T* buf_A, T* buf_B, int n){
      auto tb = SizeToBlockThreadPair(n);
      int total_size = tb.first*tb.second;
      int* d_is_equal;
      int h_is_equal = 1;
      CUDA_CHECK(cudaMalloc((void**)&d_is_equal, sizeof(int)));
      CUDA_CHECK(cudaMemcpy(d_is_equal, &h_is_equal, sizeof(int), cudaMemcpyHostToDevice));
      for(int curr_pos = 0; curr_pos < n; curr_pos += total_size){
        ker_is_equal_array_and_array<<<tb.first, tb.second>>>(buf_A+curr_pos, buf_B+curr_pos, min(total_size, n-curr_pos), d_is_equal);
      }
      CUDA_CHECK(cudaMemcpy(&h_is_equal, d_is_equal, sizeof(int), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_is_equal));
      return h_is_equal==1;
     }


   template void set_buffer_local(double *sub_buffer, const Box& box, int x, int y, int z, double val, int sw) ;
   template void set_buffer_consts(double *buffer, int n, double val);
   template void get_buffer_subarray(double *sub_buffer, double *buffer, const Box &sub_box,  const Box &box, int sw);
   template void set_buffer_seqs(double *buffer, const Shape& s, Box box, int sw);
   template void copy_buffer(double *A, double *B, int n);
   template void set_ghost_consts(double *buffer, const Shape &sp, double val, int sw);
   template ArrayPtr set_local_array(const Shape& gs, double* buf);
   template bool is_equal_arrayptr_and_array(const ArrayPtr& A, double* B);
   template bool is_equal_arrayptr_and_scalar(const ArrayPtr& A, double B);
   template bool is_equal_array_and_array(double* buf_A, double* buf_B, int n);
   template void set_buffer_local(float *sub_buffer, const Box& box, int x, int y, int z, float val, int sw) ;
   template void set_buffer_consts(float *buffer, int n, float val);
   template void get_buffer_subarray(float *sub_buffer, float *buffer, const Box &sub_box,  const Box &box, int sw);
   template void set_buffer_seqs(float *buffer, const Shape& s, Box box, int sw);
   template void copy_buffer(float *A, float *B, int n);
   template void set_ghost_consts(float *buffer, const Shape &sp, float val, int sw);
   template ArrayPtr set_local_array(const Shape& gs, float* buf);
   template bool is_equal_arrayptr_and_array(const ArrayPtr& A, float* B);
   template bool is_equal_arrayptr_and_scalar(const ArrayPtr& A, float B);
   template bool is_equal_array_and_array(float* buf_A, float* buf_B, int n);
   template void set_buffer_local(int *sub_buffer, const Box& box, int x, int y, int z, int val, int sw) ;
   template void set_buffer_consts(int *buffer, int n, int val);
   template void get_buffer_subarray(int *sub_buffer, int *buffer, const Box &sub_box,  const Box &box, int sw);
   template void set_buffer_seqs(int *buffer, const Shape& s, Box box, int sw);
   template void copy_buffer(int *A, int *B, int n);
   template void set_ghost_consts(int *buffer, const Shape &sp, int val, int sw);
   template ArrayPtr set_local_array(const Shape& gs, int* buf);
   template bool is_equal_arrayptr_and_array(const ArrayPtr& A, int* B);
   template bool is_equal_arrayptr_and_scalar(const ArrayPtr& A, int B);
   template bool is_equal_array_and_array(int* buf_A, int* buf_B, int n);

   template void  copy_buffer( double* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  double* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(double* buffer, double val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( double* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  float* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(double* buffer, float val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( double* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  int* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(double* buffer, int val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( float* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  double* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(float* buffer, double val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( float* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  float* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(float* buffer, float val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( float* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  int* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(float* buffer, int val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( int* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  double* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(int* buffer, double val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( int* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  float* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(int* buffer, float val, const Box &box, 
                  const Box &sub_box, int sw);
   template void  copy_buffer( int* A_buf, const Shape& A_buf_shape, const Box&  A_window, 
                  int* B_buf, const Shape& B_buf_shape, const Box& B_window);
   template void set_buffer_subarray_const(int* buffer, int val, const Box &box, 
                  const Box &sub_box, int sw);

  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( double* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( float* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    double* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    float* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    double* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    float* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);
  template void copy_buffer_with_mask( int* A_buf, const Shape& A_buf_shape,  const Box&  A_window,
    int* B_buf, const Shape& B_buf_shape, const Box& B_window, 
    int* C_buf, const Shape& C_buf_shape, const Box& C_window, bool ifscalar_B);

  }
}

#endif
#endif
