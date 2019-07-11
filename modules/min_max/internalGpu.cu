#ifdef __HAVE_CUDA__
#include "../../CUDA.hpp"

#define threadsPerBlock 256

namespace oa {
  namespace internal {
    namespace gpu {

 
    template<typename T>
    __global__ 
    void ker1_buffer_max_const(T *A, int* values, T *tmp_val, int *tmp_pos) {

      const int M = values[0];
      const int N = values[1];
      const int K = values[2];
      const int xs = values[3]; const int xe = values[4];
      const int ys = values[5]; const int ye = values[6];
      const int zs = values[7]; const int ze = values[8];

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      // copy from global_memory to shared_memory, per block
      if (tidInBlock < (xe-xs)) {
        int k = zs + blockIdx.x;
        int j = ys + blockIdx.y;
        int i = xs + tidInBlock;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = (A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }
      else {
        int k = zs;
        int j = ys;
        int i = xs;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = (A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }

      //__syncthreads();

      for (int idx_tmp = zs + blockIdx.x; idx_tmp < ze; idx_tmp += gridDim.x) {  
      for (int idy_tmp = ys + blockIdx.y; idy_tmp < ye; idy_tmp += gridDim.y) {  
      for (int id =      xs + tidInBlock; id < xe; id += blockDim.x) {
          int k = idx_tmp;
          int j = idy_tmp;
          int i = id;
          int dst = i + j * M + k * M * N;

          if ((A[dst]) > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = (A[dst]);
            partial_pos[tidInBlock*3   ] = i;
            partial_pos[tidInBlock*3 +1] = j;
            partial_pos[tidInBlock*3 +2] = k;
          }
      }}}

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[bid] = partial_val[0];
        tmp_pos[bid*3 + 0] = partial_pos[0];
        tmp_pos[bid*3 + 1] = partial_pos[1];
        tmp_pos[bid*3 + 2] = partial_pos[2];
      }


    }


    template<typename T>
    __global__ 
    void ker2_buffer_max_const(T *tmp_val, int *tmp_pos, const int d0, const int d1) {

      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int tid = tidInBlock;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      const int size = d0 * d1;
      // copy from global_memory to shared_memory, per block
      if (tidInBlock < size) {
        partial_val[tidInBlock] = tmp_val[tidInBlock];
        partial_pos[tidInBlock*3   ] = tmp_pos[tidInBlock*3   ];
        partial_pos[tidInBlock*3 +1] = tmp_pos[tidInBlock*3 +1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[tidInBlock*3 +2];
      }
      else {
        partial_val[tidInBlock] = tmp_val[0];
        partial_pos[tidInBlock*3   ] = tmp_pos[0];
        partial_pos[tidInBlock*3 +1] = tmp_pos[1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[2];
      }

      //__syncthreads();

      for (int id = tidInBlock + blockDim.x; id < size; id += blockDim.x) {
          if (tmp_val[id] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = tmp_val[id];
            partial_pos[tidInBlock*3   ] = tmp_pos[id*3   ];
            partial_pos[tidInBlock*3 +1] = tmp_pos[id*3 +1];
            partial_pos[tidInBlock*3 +2] = tmp_pos[id*3 +2];
          }
      }

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[0] = partial_val[0];
        tmp_pos[0] = partial_pos[0];
        tmp_pos[1] = partial_pos[1];
        tmp_pos[2] = partial_pos[2];
      }


    }
    
    template<typename T>
    void ker_buffer_max_const(T *A, int* host_values, T &val, int *pos) {
      
      const int xs = host_values[3]; const int xe = host_values[4];
      const int ys = host_values[5]; const int ye = host_values[6];
      const int zs = host_values[7]; const int ze = host_values[8];
      
      const unsigned int dimBlock0 = ze-zs;
      const unsigned int dimBlock1 = ye-ys;
      
      const unsigned int size = sizeof(int)*9;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);
      T* dev_tmp_val;
      cudaMalloc((void**)&dev_tmp_val, sizeof(T)*dimBlock0*dimBlock1);
      int* dev_tmp_pos;
      cudaMalloc((void**)&dev_tmp_pos, sizeof(int)*dimBlock0*dimBlock1*3);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_max_const<<<dimBlock, dimThread>>>(A, dev_values, dev_tmp_val, dev_tmp_pos);

      ker2_buffer_max_const<<<1, dimThread>>>(dev_tmp_val, dev_tmp_pos, dimBlock0, dimBlock1);

      T tmp_val;
      cudaMemcpy(&tmp_val, dev_tmp_val, sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy(pos, dev_tmp_pos, sizeof(int)*3, cudaMemcpyDeviceToHost);

      val = tmp_val;

      cudaFree(dev_values);
      cudaFree(dev_tmp_val);
      cudaFree(dev_tmp_pos);


    }


 
    template<typename T>
    __global__ 
    void ker1_buffer_min_const(T *A, int* values, T *tmp_val, int *tmp_pos) {

      const int M = values[0];
      const int N = values[1];
      const int K = values[2];
      const int xs = values[3]; const int xe = values[4];
      const int ys = values[5]; const int ye = values[6];
      const int zs = values[7]; const int ze = values[8];

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      // copy from global_memory to shared_memory, per block
      if (tidInBlock < (xe-xs)) {
        int k = zs + blockIdx.x;
        int j = ys + blockIdx.y;
        int i = xs + tidInBlock;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = (A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }
      else {
        int k = zs;
        int j = ys;
        int i = xs;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = (A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }

      //__syncthreads();

      for (int idx_tmp = zs + blockIdx.x; idx_tmp < ze; idx_tmp += gridDim.x) {  
      for (int idy_tmp = ys + blockIdx.y; idy_tmp < ye; idy_tmp += gridDim.y) {  
      for (int id =      xs + tidInBlock; id < xe; id += blockDim.x) {
          int k = idx_tmp;
          int j = idy_tmp;
          int i = id;
          int dst = i + j * M + k * M * N;

          if ((A[dst]) < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = (A[dst]);
            partial_pos[tidInBlock*3   ] = i;
            partial_pos[tidInBlock*3 +1] = j;
            partial_pos[tidInBlock*3 +2] = k;
          }
      }}}

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[bid] = partial_val[0];
        tmp_pos[bid*3 + 0] = partial_pos[0];
        tmp_pos[bid*3 + 1] = partial_pos[1];
        tmp_pos[bid*3 + 2] = partial_pos[2];
      }


    }


    template<typename T>
    __global__ 
    void ker2_buffer_min_const(T *tmp_val, int *tmp_pos, const int d0, const int d1) {

      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int tid = tidInBlock;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      const int size = d0 * d1;
      // copy from global_memory to shared_memory, per block
      if (tidInBlock < size) {
        partial_val[tidInBlock] = tmp_val[tidInBlock];
        partial_pos[tidInBlock*3   ] = tmp_pos[tidInBlock*3   ];
        partial_pos[tidInBlock*3 +1] = tmp_pos[tidInBlock*3 +1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[tidInBlock*3 +2];
      }
      else {
        partial_val[tidInBlock] = tmp_val[0];
        partial_pos[tidInBlock*3   ] = tmp_pos[0];
        partial_pos[tidInBlock*3 +1] = tmp_pos[1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[2];
      }

      //__syncthreads();

      for (int id = tidInBlock + blockDim.x; id < size; id += blockDim.x) {
          if (tmp_val[id] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = tmp_val[id];
            partial_pos[tidInBlock*3   ] = tmp_pos[id*3   ];
            partial_pos[tidInBlock*3 +1] = tmp_pos[id*3 +1];
            partial_pos[tidInBlock*3 +2] = tmp_pos[id*3 +2];
          }
      }

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[0] = partial_val[0];
        tmp_pos[0] = partial_pos[0];
        tmp_pos[1] = partial_pos[1];
        tmp_pos[2] = partial_pos[2];
      }


    }
    
    template<typename T>
    void ker_buffer_min_const(T *A, int* host_values, T &val, int *pos) {
      
      const int xs = host_values[3]; const int xe = host_values[4];
      const int ys = host_values[5]; const int ye = host_values[6];
      const int zs = host_values[7]; const int ze = host_values[8];
      
      const unsigned int dimBlock0 = ze-zs;
      const unsigned int dimBlock1 = ye-ys;
      
      const unsigned int size = sizeof(int)*9;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);
      T* dev_tmp_val;
      cudaMalloc((void**)&dev_tmp_val, sizeof(T)*dimBlock0*dimBlock1);
      int* dev_tmp_pos;
      cudaMalloc((void**)&dev_tmp_pos, sizeof(int)*dimBlock0*dimBlock1*3);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_min_const<<<dimBlock, dimThread>>>(A, dev_values, dev_tmp_val, dev_tmp_pos);

      ker2_buffer_min_const<<<1, dimThread>>>(dev_tmp_val, dev_tmp_pos, dimBlock0, dimBlock1);

      T tmp_val;
      cudaMemcpy(&tmp_val, dev_tmp_val, sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy(pos, dev_tmp_pos, sizeof(int)*3, cudaMemcpyDeviceToHost);

      val = tmp_val;

      cudaFree(dev_values);
      cudaFree(dev_tmp_val);
      cudaFree(dev_tmp_pos);


    }


 
    template<typename T>
    __global__ 
    void ker1_buffer_abs_max_const(T *A, int* values, T *tmp_val, int *tmp_pos) {

      const int M = values[0];
      const int N = values[1];
      const int K = values[2];
      const int xs = values[3]; const int xe = values[4];
      const int ys = values[5]; const int ye = values[6];
      const int zs = values[7]; const int ze = values[8];

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      // copy from global_memory to shared_memory, per block
      if (tidInBlock < (xe-xs)) {
        int k = zs + blockIdx.x;
        int j = ys + blockIdx.y;
        int i = xs + tidInBlock;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = std::abs(A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }
      else {
        int k = zs;
        int j = ys;
        int i = xs;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = std::abs(A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }

      //__syncthreads();

      for (int idx_tmp = zs + blockIdx.x; idx_tmp < ze; idx_tmp += gridDim.x) {  
      for (int idy_tmp = ys + blockIdx.y; idy_tmp < ye; idy_tmp += gridDim.y) {  
      for (int id =      xs + tidInBlock; id < xe; id += blockDim.x) {
          int k = idx_tmp;
          int j = idy_tmp;
          int i = id;
          int dst = i + j * M + k * M * N;

          if (std::abs(A[dst]) > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = std::abs(A[dst]);
            partial_pos[tidInBlock*3   ] = i;
            partial_pos[tidInBlock*3 +1] = j;
            partial_pos[tidInBlock*3 +2] = k;
          }
      }}}

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[bid] = partial_val[0];
        tmp_pos[bid*3 + 0] = partial_pos[0];
        tmp_pos[bid*3 + 1] = partial_pos[1];
        tmp_pos[bid*3 + 2] = partial_pos[2];
      }


    }


    template<typename T>
    __global__ 
    void ker2_buffer_abs_max_const(T *tmp_val, int *tmp_pos, const int d0, const int d1) {

      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int tid = tidInBlock;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      const int size = d0 * d1;
      // copy from global_memory to shared_memory, per block
      if (tidInBlock < size) {
        partial_val[tidInBlock] = tmp_val[tidInBlock];
        partial_pos[tidInBlock*3   ] = tmp_pos[tidInBlock*3   ];
        partial_pos[tidInBlock*3 +1] = tmp_pos[tidInBlock*3 +1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[tidInBlock*3 +2];
      }
      else {
        partial_val[tidInBlock] = tmp_val[0];
        partial_pos[tidInBlock*3   ] = tmp_pos[0];
        partial_pos[tidInBlock*3 +1] = tmp_pos[1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[2];
      }

      //__syncthreads();

      for (int id = tidInBlock + blockDim.x; id < size; id += blockDim.x) {
          if (tmp_val[id] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = tmp_val[id];
            partial_pos[tidInBlock*3   ] = tmp_pos[id*3   ];
            partial_pos[tidInBlock*3 +1] = tmp_pos[id*3 +1];
            partial_pos[tidInBlock*3 +2] = tmp_pos[id*3 +2];
          }
      }

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] > partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[0] = partial_val[0];
        tmp_pos[0] = partial_pos[0];
        tmp_pos[1] = partial_pos[1];
        tmp_pos[2] = partial_pos[2];
      }


    }
    
    template<typename T>
    void ker_buffer_abs_max_const(T *A, int* host_values, T &val, int *pos) {
      
      const int xs = host_values[3]; const int xe = host_values[4];
      const int ys = host_values[5]; const int ye = host_values[6];
      const int zs = host_values[7]; const int ze = host_values[8];
      
      const unsigned int dimBlock0 = ze-zs;
      const unsigned int dimBlock1 = ye-ys;
      
      const unsigned int size = sizeof(int)*9;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);
      T* dev_tmp_val;
      cudaMalloc((void**)&dev_tmp_val, sizeof(T)*dimBlock0*dimBlock1);
      int* dev_tmp_pos;
      cudaMalloc((void**)&dev_tmp_pos, sizeof(int)*dimBlock0*dimBlock1*3);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_abs_max_const<<<dimBlock, dimThread>>>(A, dev_values, dev_tmp_val, dev_tmp_pos);

      ker2_buffer_abs_max_const<<<1, dimThread>>>(dev_tmp_val, dev_tmp_pos, dimBlock0, dimBlock1);

      T tmp_val;
      cudaMemcpy(&tmp_val, dev_tmp_val, sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy(pos, dev_tmp_pos, sizeof(int)*3, cudaMemcpyDeviceToHost);

      val = tmp_val;

      cudaFree(dev_values);
      cudaFree(dev_tmp_val);
      cudaFree(dev_tmp_pos);


    }


 
    template<typename T>
    __global__ 
    void ker1_buffer_abs_min_const(T *A, int* values, T *tmp_val, int *tmp_pos) {

      const int M = values[0];
      const int N = values[1];
      const int K = values[2];
      const int xs = values[3]; const int xe = values[4];
      const int ys = values[5]; const int ye = values[6];
      const int zs = values[7]; const int ze = values[8];

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      // copy from global_memory to shared_memory, per block
      if (tidInBlock < (xe-xs)) {
        int k = zs + blockIdx.x;
        int j = ys + blockIdx.y;
        int i = xs + tidInBlock;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = std::abs(A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }
      else {
        int k = zs;
        int j = ys;
        int i = xs;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = std::abs(A[dst]);
        partial_pos[tidInBlock*3   ] = i;
        partial_pos[tidInBlock*3 +1] = j;
        partial_pos[tidInBlock*3 +2] = k;
      }

      //__syncthreads();

      for (int idx_tmp = zs + blockIdx.x; idx_tmp < ze; idx_tmp += gridDim.x) {  
      for (int idy_tmp = ys + blockIdx.y; idy_tmp < ye; idy_tmp += gridDim.y) {  
      for (int id =      xs + tidInBlock; id < xe; id += blockDim.x) {
          int k = idx_tmp;
          int j = idy_tmp;
          int i = id;
          int dst = i + j * M + k * M * N;

          if (std::abs(A[dst]) < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = std::abs(A[dst]);
            partial_pos[tidInBlock*3   ] = i;
            partial_pos[tidInBlock*3 +1] = j;
            partial_pos[tidInBlock*3 +2] = k;
          }
      }}}

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[bid] = partial_val[0];
        tmp_pos[bid*3 + 0] = partial_pos[0];
        tmp_pos[bid*3 + 1] = partial_pos[1];
        tmp_pos[bid*3 + 2] = partial_pos[2];
      }


    }


    template<typename T>
    __global__ 
    void ker2_buffer_abs_min_const(T *tmp_val, int *tmp_pos, const int d0, const int d1) {

      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int tid = tidInBlock;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      const int size = d0 * d1;
      // copy from global_memory to shared_memory, per block
      if (tidInBlock < size) {
        partial_val[tidInBlock] = tmp_val[tidInBlock];
        partial_pos[tidInBlock*3   ] = tmp_pos[tidInBlock*3   ];
        partial_pos[tidInBlock*3 +1] = tmp_pos[tidInBlock*3 +1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[tidInBlock*3 +2];
      }
      else {
        partial_val[tidInBlock] = tmp_val[0];
        partial_pos[tidInBlock*3   ] = tmp_pos[0];
        partial_pos[tidInBlock*3 +1] = tmp_pos[1];
        partial_pos[tidInBlock*3 +2] = tmp_pos[2];
      }

      //__syncthreads();

      for (int id = tidInBlock + blockDim.x; id < size; id += blockDim.x) {
          if (tmp_val[id] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = tmp_val[id];
            partial_pos[tidInBlock*3   ] = tmp_pos[id*3   ];
            partial_pos[tidInBlock*3 +1] = tmp_pos[id*3 +1];
            partial_pos[tidInBlock*3 +2] = tmp_pos[id*3 +2];
          }
      }

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
          if (partial_val[tidInBlock+stride] < partial_val[tidInBlock]) {
            partial_val[tidInBlock] = partial_val[tidInBlock+stride];
            partial_pos[tidInBlock*3   ] = partial_pos[(tidInBlock+stride)*3   ];
            partial_pos[tidInBlock*3 +1] = partial_pos[(tidInBlock+stride)*3 +1];
            partial_pos[tidInBlock*3 +2] = partial_pos[(tidInBlock+stride)*3 +2];
          }
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[0] = partial_val[0];
        tmp_pos[0] = partial_pos[0];
        tmp_pos[1] = partial_pos[1];
        tmp_pos[2] = partial_pos[2];
      }


    }
    
    template<typename T>
    void ker_buffer_abs_min_const(T *A, int* host_values, T &val, int *pos) {
      
      const int xs = host_values[3]; const int xe = host_values[4];
      const int ys = host_values[5]; const int ye = host_values[6];
      const int zs = host_values[7]; const int ze = host_values[8];
      
      const unsigned int dimBlock0 = ze-zs;
      const unsigned int dimBlock1 = ye-ys;
      
      const unsigned int size = sizeof(int)*9;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);
      T* dev_tmp_val;
      cudaMalloc((void**)&dev_tmp_val, sizeof(T)*dimBlock0*dimBlock1);
      int* dev_tmp_pos;
      cudaMalloc((void**)&dev_tmp_pos, sizeof(int)*dimBlock0*dimBlock1*3);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_abs_min_const<<<dimBlock, dimThread>>>(A, dev_values, dev_tmp_val, dev_tmp_pos);

      ker2_buffer_abs_min_const<<<1, dimThread>>>(dev_tmp_val, dev_tmp_pos, dimBlock0, dimBlock1);

      T tmp_val;
      cudaMemcpy(&tmp_val, dev_tmp_val, sizeof(T), cudaMemcpyDeviceToHost);
      cudaMemcpy(pos, dev_tmp_pos, sizeof(int)*3, cudaMemcpyDeviceToHost);

      val = tmp_val;

      cudaFree(dev_values);
      cudaFree(dev_tmp_val);
      cudaFree(dev_tmp_pos);


    }





    template<typename TC, typename TA, typename TB>
    __global__
    void ker1_buffer_max2(TC* C, TA *A, TB *B, int* dev_values) {
      const int xs_a = dev_values[0];
      const int xe_a = dev_values[1];
      const int ys_a = dev_values[2];
      const int ye_a = dev_values[3];
      const int zs_a = dev_values[4];
      const int ze_a = dev_values[5];
      const int xs_b = dev_values[6];
      const int xe_b = dev_values[7];
      const int ys_b = dev_values[8];
      const int ye_b = dev_values[9];
      const int zs_b = dev_values[10];
      const int ze_b = dev_values[11];
      const int xs_c = dev_values[12];
      const int xe_c = dev_values[13];
      const int ys_c = dev_values[14];
      const int ye_c = dev_values[15];
      const int zs_c = dev_values[16];
      const int ze_c = dev_values[17];

      const int MA = dev_values[18];
      const int NA = dev_values[19];
      const int PA = dev_values[20];
      const int MB = dev_values[21];
      const int NB = dev_values[22];
      const int PB = dev_values[23];
      const int MC = dev_values[24];
      const int NC = dev_values[25];
      const int PC = dev_values[26];

      const int M = xe_a - xs_a;
      const int N = ye_a - ys_a;
      const int P = ze_a - zs_a;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

      if (tidInBlock >= M) return; 

      for(int k = blockIdx.x; k < P; k += gridDim.x){
        for(int j = blockIdx.y; j < N; j += gridDim.y){
          for(int i = tidInBlock; i < M; i += blockDim.x){
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
    void ker_buffer_max2(TC* C, TA *A, TB *B, int* host_values){
      
      const int M = host_values[1] - host_values[0];
      const int N = host_values[3] - host_values[2];
      const int P = host_values[5] - host_values[4];
      const unsigned int size = sizeof(int)*27;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);

      const unsigned int dimBlock0 = P;
      const unsigned int dimBlock1 = N;
      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_max2<<<dimBlock, dimThread>>>(C, A, B, dev_values);

      cudaFree(dev_values);

    }

    template<typename TC, typename TA, typename TB>
    __global__
    void ker1_buffer_min2(TC* C, TA *A, TB *B, int* dev_values) {
      const int xs_a = dev_values[0];
      const int xe_a = dev_values[1];
      const int ys_a = dev_values[2];
      const int ye_a = dev_values[3];
      const int zs_a = dev_values[4];
      const int ze_a = dev_values[5];
      const int xs_b = dev_values[6];
      const int xe_b = dev_values[7];
      const int ys_b = dev_values[8];
      const int ye_b = dev_values[9];
      const int zs_b = dev_values[10];
      const int ze_b = dev_values[11];
      const int xs_c = dev_values[12];
      const int xe_c = dev_values[13];
      const int ys_c = dev_values[14];
      const int ye_c = dev_values[15];
      const int zs_c = dev_values[16];
      const int ze_c = dev_values[17];

      const int MA = dev_values[18];
      const int NA = dev_values[19];
      const int PA = dev_values[20];
      const int MB = dev_values[21];
      const int NB = dev_values[22];
      const int PB = dev_values[23];
      const int MC = dev_values[24];
      const int NC = dev_values[25];
      const int PC = dev_values[26];

      const int M = xe_a - xs_a;
      const int N = ye_a - ys_a;
      const int P = ze_a - zs_a;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

      if (tidInBlock >= M) return; 

      for(int k = blockIdx.x; k < P; k += gridDim.x){
        for(int j = blockIdx.y; j < N; j += gridDim.y){
          for(int i = tidInBlock; i < M; i += blockDim.x){
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
    template<typename TC, typename TA, typename TB>
    void ker_buffer_min2(TC* C, TA *A, TB *B, int* host_values){
      
      const int M = host_values[1] - host_values[0];
      const int N = host_values[3] - host_values[2];
      const int P = host_values[5] - host_values[4];
      const unsigned int size = sizeof(int)*27;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size);
      cudaMemcpy(dev_values, host_values, size, cudaMemcpyHostToDevice);

      const unsigned int dimBlock0 = P;
      const unsigned int dimBlock1 = N;
      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_min2<<<dimBlock, dimThread>>>(C, A, B, dev_values);

      cudaFree(dev_values);

    }



    template void ker_buffer_max_const<int>(int *A, int* host_values, int &val, int *pos);
    template void ker_buffer_max_const<float>(float *A, int* host_values, float &val, int *pos);
    template void ker_buffer_max_const<double>(double *A, int* host_values, double &val, int *pos);
    template void ker_buffer_min_const<int>(int *A, int* host_values, int &val, int *pos);
    template void ker_buffer_min_const<float>(float *A, int* host_values, float &val, int *pos);
    template void ker_buffer_min_const<double>(double *A, int* host_values, double &val, int *pos);
    template void ker_buffer_abs_max_const<int>(int *A, int* host_values, int &val, int *pos);
    template void ker_buffer_abs_max_const<float>(float *A, int* host_values, float &val, int *pos);
    template void ker_buffer_abs_max_const<double>(double *A, int* host_values, double &val, int *pos);
    template void ker_buffer_abs_min_const<int>(int *A, int* host_values, int &val, int *pos);
    template void ker_buffer_abs_min_const<float>(float *A, int* host_values, float &val, int *pos);
    template void ker_buffer_abs_min_const<double>(double *A, int* host_values, double &val, int *pos);
    
    
    template void ker_buffer_max2<int, int, int>(int* C, int *A, int *B, int* host_values);
    template void ker_buffer_max2<float, int, int>(float* C, int *A, int *B, int* host_values);
    template void ker_buffer_max2<double, int, int>(double* C, int *A, int *B, int* host_values);
    template void ker_buffer_max2<int, int, float>(int* C, int *A, float *B, int* host_values);
    template void ker_buffer_max2<float, int, float>(float* C, int *A, float *B, int* host_values);
    template void ker_buffer_max2<double, int, float>(double* C, int *A, float *B, int* host_values);
    template void ker_buffer_max2<int, int, double>(int* C, int *A, double *B, int* host_values);
    template void ker_buffer_max2<float, int, double>(float* C, int *A, double *B, int* host_values);
    template void ker_buffer_max2<double, int, double>(double* C, int *A, double *B, int* host_values);
    template void ker_buffer_max2<int, float, int>(int* C, float *A, int *B, int* host_values);
    template void ker_buffer_max2<float, float, int>(float* C, float *A, int *B, int* host_values);
    template void ker_buffer_max2<double, float, int>(double* C, float *A, int *B, int* host_values);
    template void ker_buffer_max2<int, float, float>(int* C, float *A, float *B, int* host_values);
    template void ker_buffer_max2<float, float, float>(float* C, float *A, float *B, int* host_values);
    template void ker_buffer_max2<double, float, float>(double* C, float *A, float *B, int* host_values);
    template void ker_buffer_max2<int, float, double>(int* C, float *A, double *B, int* host_values);
    template void ker_buffer_max2<float, float, double>(float* C, float *A, double *B, int* host_values);
    template void ker_buffer_max2<double, float, double>(double* C, float *A, double *B, int* host_values);
    template void ker_buffer_max2<int, double, int>(int* C, double *A, int *B, int* host_values);
    template void ker_buffer_max2<float, double, int>(float* C, double *A, int *B, int* host_values);
    template void ker_buffer_max2<double, double, int>(double* C, double *A, int *B, int* host_values);
    template void ker_buffer_max2<int, double, float>(int* C, double *A, float *B, int* host_values);
    template void ker_buffer_max2<float, double, float>(float* C, double *A, float *B, int* host_values);
    template void ker_buffer_max2<double, double, float>(double* C, double *A, float *B, int* host_values);
    template void ker_buffer_max2<int, double, double>(int* C, double *A, double *B, int* host_values);
    template void ker_buffer_max2<float, double, double>(float* C, double *A, double *B, int* host_values);
    template void ker_buffer_max2<double, double, double>(double* C, double *A, double *B, int* host_values);
    template void ker_buffer_min2<int, int, int>(int* C, int *A, int *B, int* host_values);
    template void ker_buffer_min2<float, int, int>(float* C, int *A, int *B, int* host_values);
    template void ker_buffer_min2<double, int, int>(double* C, int *A, int *B, int* host_values);
    template void ker_buffer_min2<int, int, float>(int* C, int *A, float *B, int* host_values);
    template void ker_buffer_min2<float, int, float>(float* C, int *A, float *B, int* host_values);
    template void ker_buffer_min2<double, int, float>(double* C, int *A, float *B, int* host_values);
    template void ker_buffer_min2<int, int, double>(int* C, int *A, double *B, int* host_values);
    template void ker_buffer_min2<float, int, double>(float* C, int *A, double *B, int* host_values);
    template void ker_buffer_min2<double, int, double>(double* C, int *A, double *B, int* host_values);
    template void ker_buffer_min2<int, float, int>(int* C, float *A, int *B, int* host_values);
    template void ker_buffer_min2<float, float, int>(float* C, float *A, int *B, int* host_values);
    template void ker_buffer_min2<double, float, int>(double* C, float *A, int *B, int* host_values);
    template void ker_buffer_min2<int, float, float>(int* C, float *A, float *B, int* host_values);
    template void ker_buffer_min2<float, float, float>(float* C, float *A, float *B, int* host_values);
    template void ker_buffer_min2<double, float, float>(double* C, float *A, float *B, int* host_values);
    template void ker_buffer_min2<int, float, double>(int* C, float *A, double *B, int* host_values);
    template void ker_buffer_min2<float, float, double>(float* C, float *A, double *B, int* host_values);
    template void ker_buffer_min2<double, float, double>(double* C, float *A, double *B, int* host_values);
    template void ker_buffer_min2<int, double, int>(int* C, double *A, int *B, int* host_values);
    template void ker_buffer_min2<float, double, int>(float* C, double *A, int *B, int* host_values);
    template void ker_buffer_min2<double, double, int>(double* C, double *A, int *B, int* host_values);
    template void ker_buffer_min2<int, double, float>(int* C, double *A, float *B, int* host_values);
    template void ker_buffer_min2<float, double, float>(float* C, double *A, float *B, int* host_values);
    template void ker_buffer_min2<double, double, float>(double* C, double *A, float *B, int* host_values);
    template void ker_buffer_min2<int, double, double>(int* C, double *A, double *B, int* host_values);
    template void ker_buffer_min2<float, double, double>(float* C, double *A, double *B, int* host_values);
    template void ker_buffer_min2<double, double, double>(double* C, double *A, double *B, int* host_values);
    }
  }
}

#endif
