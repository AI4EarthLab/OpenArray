#ifdef __HAVE_CUDA__
#include "../../CUDA.hpp"

#define threadsPerBlock 256

namespace oa {
namespace internal {
namespace gpu {

    template<typename T>
    __global__ 
    void ker1_buffer_sum_scalar_const(T *A, const int *values, const int sw, const int size, T *tmp_val) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;
      const unsigned int tid = idx + idy * blockDim.x * gridDim.x;
      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;

     
      __shared__ T    partial_val[threadsPerBlock];
      __shared__ int  partial_pos[threadsPerBlock*3];

      // copy from global_memory to shared_memory, per block
      //if (tidInBlock < (xe-xs)) 
      {
        int k = sw + blockIdx.x;
        int j = sw + blockIdx.y;
        int i = sw + tidInBlock;
        int dst = i + j * M + k * M * N;

        partial_val[tidInBlock] = (T)0.0;
      }

      //__syncthreads();

      for (int idx_tmp = sw + blockIdx.x; idx_tmp < K-sw; idx_tmp += gridDim.x) {  
      for (int idy_tmp = sw + blockIdx.y; idy_tmp < N-sw; idy_tmp += gridDim.y) {  
      for (int id =      sw + tidInBlock; id < M-sw; id += blockDim.x) {
          int k = idx_tmp;
          int j = idy_tmp;
          int i = id;
          int dst = i + j * M + k * M * N;

          partial_val[tidInBlock] += A[dst];
      }}}

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
            partial_val[tidInBlock] += partial_val[tidInBlock+stride];
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[bid] = partial_val[0];
      }


    }


    template<typename T>
    __global__ 
    void ker2_buffer_sum_scalar_const(T *tmp_val, const int dimBlock0, const int dimBlock1) {

      const unsigned int tidInBlock = threadIdx.x;
      const unsigned int tid = tidInBlock;

     
      __shared__ T    partial_val[threadsPerBlock];

      const int size = dimBlock0 * dimBlock1;
      // copy from global_memory to shared_memory, per block
      if (tidInBlock < size) {
        partial_val[tidInBlock] = tmp_val[tidInBlock];
      }
      else {
        partial_val[tidInBlock] = (T)0.0;
      }

      //__syncthreads();

      for (int id = tidInBlock + blockDim.x; id < size; id += blockDim.x) {
            partial_val[tidInBlock] += tmp_val[id];
      }

      //__syncthreads();

      for (int stride = (threadsPerBlock >> 1); stride > 0; stride >>= 1) {
        if (tidInBlock < stride) {
            partial_val[tidInBlock] += partial_val[tidInBlock+stride];
        }

        //__syncthreads();
      }

      if (tidInBlock == 0) {
        tmp_val[0] = partial_val[0];
      }


    }
    
    template<typename T>
    void ker_buffer_sum_scalar_const(T *val, T *A, const int *host_values, const int sw, const int size) {
      
      const int xs = host_values[0]; const int xe = host_values[1];
      const int ys = host_values[2]; const int ye = host_values[3];
      const int zs = host_values[4]; const int ze = host_values[5];
      
      const unsigned int dimBlock0 = ze-zs;
      const unsigned int dimBlock1 = ye-ys;
      
      const unsigned int size_of_values = sizeof(int)*6;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size_of_values);
      cudaMemcpy(dev_values, host_values, size_of_values, cudaMemcpyHostToDevice);
      T* dev_tmp_val;
      cudaMalloc((void**)&dev_tmp_val, sizeof(T)*dimBlock0*dimBlock1);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);
      ker1_buffer_sum_scalar_const<<<dimBlock, dimThread>>>(A, dev_values, sw, size, dev_tmp_val);

      ker2_buffer_sum_scalar_const<<<1, dimThread>>>(dev_tmp_val, dimBlock0, dimBlock1);

      T tmp_val;
      cudaMemcpy(&tmp_val, dev_tmp_val, sizeof(T), cudaMemcpyDeviceToHost);

      *val = tmp_val;

      cudaFree(dev_values);
      cudaFree(dev_tmp_val);

    }

    template void ker_buffer_sum_scalar_const<int>(int *val, int *A, const int *host_values, const int sw, const int size);
    template void ker_buffer_sum_scalar_const<float>(float *val, float *A, const int *host_values, const int sw, const int size);
    template void ker_buffer_sum_scalar_const<double>(double *val, double *A, const int *host_values, const int sw, const int size);
    

    template<typename T>
    __global__
    void ker1_buffer_csum_z_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      if(type == 2) {
        for (int j = idx; j < N - 2*sw; j += blockDim.x*gridDim.x) {
        for (int i = idy; i < M - 2*sw; i += blockDim.y*gridDim.y) {
          int index = j*(M-2*sw) + i;
          buffer[index] = 0;
        }}
      }

      for (int k = sw; k < K - sw; k++) {
	      if((k == sw) && (type == 1 || type == 0)) {
	        for (int j = sw + idx; j < N - sw; j += blockDim.x*gridDim.x) {
	        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          int index = (M-2*sw)*(j-sw) + (i-sw);
	          ap[temp1] = buffer[index];
	        }}
	      } else {
	        for (int j = sw + idx; j < N - sw; j += blockDim.x*gridDim.x) {
	        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          ap[temp1] = 0;
	        }}
	      }
      }

    }

    template<typename T>
    __global__
    void ker2_buffer_csum_z_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type, const int z) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      const int k = z;

      if(type == 1 || type == 2){
        if(k < K - sw - 1)
        for (int j = sw + idx; j < N - sw; j += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M*N;
          ap[temp1]+=A[temp1];
          //if(k < K - sw - 1)
            ap[temp2] += ap[temp1];
        }}
        
        if(k == K - sw - 1 )
        for (int j = sw + idx; j < N - sw; j += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M*N;
          int index = (M-2*sw)*(j-sw) + (i-sw);
          ap[temp1]+=A[temp1];
          //if(k == K - sw - 1 )
            buffer[index] = ap[temp1];
        }}
      } else {
        //if(k < K - sw - 1)
        for (int j = sw + idx; j < N - sw; j += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M*N;
          ap[temp1]+=A[temp1];
          if(k < K - sw - 1)
            ap[temp2] += ap[temp1];
        }}
      }

    }

    template<typename T>
    void ker_buffer_csum_z_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type) {

      const int xs = host_values[0]; const int xe = host_values[1];
      const int ys = host_values[2]; const int ye = host_values[3];
      const int zs = host_values[4]; const int ze = host_values[5];
      
      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int dimBlock0 = ye;
      const unsigned int dimBlock1 = ze;
      
      const unsigned int size_of_values = sizeof(int)*6;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size_of_values);
      cudaMemcpy(dev_values, host_values, size_of_values, cudaMemcpyHostToDevice);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);

      ker1_buffer_csum_z_const<<<dimBlock, dimThread>>>(ap, A, buffer, dev_values, sw, size, type);
      cudaDeviceSynchronize();


      dim3 dimBlock2(dimBlock0, dimBlock1, 1);
      dim3 dimThread2(1, 1, 1);
      for (int z = sw; z < K - sw; z++) {
        ker2_buffer_csum_z_const<<<dimBlock2, dimThread2>>>(ap, A, buffer, dev_values, sw, size, type, z);
        cudaDeviceSynchronize();
      }

      cudaFree(dev_values);

    }

    template void ker_buffer_csum_z_const<int>(
        int *ap, int *A, int *host_values, 
        int sw, int size, int *buffer, int type);
    template void ker_buffer_csum_z_const<float>(
        float *ap, float *A, int *host_values, 
        int sw, int size, float *buffer, int type);
    template void ker_buffer_csum_z_const<double>(
        double *ap, double *A, int *host_values, 
        int sw, int size, double *buffer, int type);



    template<typename T>
    __global__
    void ker1_buffer_csum_y_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      if(type == 2) {
        for (int k = idx; k < K - 2*sw; k += blockDim.x*gridDim.x) {
        for (int i = idy; i < M - 2*sw; i += blockDim.y*gridDim.y) {
          int index = k*(M-2*sw) + i;
          buffer[index] = 0;
        }}
      }

      for (int j = sw; j < N - sw; j++) {
	      if((j == sw) && (type == 1 || type == 0)) {
	        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
	        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          int index = (M-2*sw)*(k-sw) + (i-sw);
	          ap[temp1] = buffer[index];
	        }}
	      } else {
	        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
	        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          ap[temp1] = 0;
	        }}
	      }
      }

    }

    template<typename T>
    __global__
    void ker2_buffer_csum_y_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type, const int y) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      const int j = y;

      if(type == 1 || type == 2){
        if(j < N - sw - 1)
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M;
          ap[temp1]+=A[temp1];
          //if(k < K - sw - 1)
            ap[temp2] += ap[temp1];
        }}
        
        if(j == N - sw - 1 )
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M;
          int index = (M-2*sw)*(k-sw) + (i-sw);
          ap[temp1]+=A[temp1];
          //if(k == K - sw - 1 )
            buffer[index] = ap[temp1];
        }}
      } else {
        //if(j < N - sw - 1)
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int i = sw + idy; i < M - sw; i += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + M;
          ap[temp1]+=A[temp1];
          if(j < N - sw - 1)
            ap[temp2] += ap[temp1];
        }}
      }

    }

    template<typename T>
    void ker_buffer_csum_y_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type) {

      const int xs = host_values[0]; const int xe = host_values[1];
      const int ys = host_values[2]; const int ye = host_values[3];
      const int zs = host_values[4]; const int ze = host_values[5];
      
      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int dimBlock0 = ze;
      const unsigned int dimBlock1 = xe;
      
      const unsigned int size_of_values = sizeof(int)*6;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size_of_values);
      cudaMemcpy(dev_values, host_values, size_of_values, cudaMemcpyHostToDevice);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);

      ker1_buffer_csum_y_const<<<dimBlock, dimThread>>>(ap, A, buffer, dev_values, sw, size, type);
      cudaDeviceSynchronize();


      dim3 dimBlock2(dimBlock0, dimBlock1, 1);
      dim3 dimThread2(1, 1, 1);
      for (int y = sw; y < N - sw; y++) {
        ker2_buffer_csum_y_const<<<dimBlock2, dimThread2>>>(ap, A, buffer, dev_values, sw, size, type, y);
        cudaDeviceSynchronize();
      }

      cudaFree(dev_values);

    }

    template void ker_buffer_csum_y_const<int>(
        int *ap, int *A, int *host_values, 
        int sw, int size, int *buffer, int type);
    template void ker_buffer_csum_y_const<float>(
        float *ap, float *A, int *host_values, 
        int sw, int size, float *buffer, int type);
    template void ker_buffer_csum_y_const<double>(
        double *ap, double *A, int *host_values, 
        int sw, int size, double *buffer, int type);


    template<typename T>
    __global__
    void ker1_buffer_csum_x_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      if(type == 2) {
        for (int k = idx; k < K - 2*sw; k += blockDim.x*gridDim.x) {
        for (int j = idy; j < N - 2*sw; j += blockDim.y*gridDim.y) {
          int index = k*(N-2*sw) + j;
          buffer[index] = 0;
        }}
      }

      for (int i = sw; i < M - sw; i++) {
	      if((i == sw) && (type == 1 || type == 0)) {
	        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
	        for (int j = sw + idy; j < N - sw; j += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          int index = (N-2*sw)*(k-sw) + (j-sw);
	          ap[temp1] = buffer[index];
	        }}
	      } else {
	        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
	        for (int j = sw + idy; j < N - sw; j += blockDim.y*gridDim.y) {
	          int temp1 = N*M*k + M*j + i;
	          ap[temp1] = 0;
	        }}
	      }
      }

    }

    template<typename T>
    __global__
    void ker2_buffer_csum_x_const(T *ap, T *A, T *buffer, int *values, int sw, int size, int type, const int x) {

      const int xs = values[0]; const int xe = values[1];
      const int ys = values[2]; const int ye = values[3];
      const int zs = values[4]; const int ze = values[5];

      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
      const unsigned int idy = (blockIdx.y * blockDim.y) + threadIdx.y;

      const int i = x;

      if(type == 1 || type == 2){
        if(i < M - sw - 1)
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int j = sw + idy; j < N - sw; j += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + 1;
          ap[temp1]+=A[temp1];
          //if(k < K - sw - 1)
            ap[temp2] += ap[temp1];
        }}
        
        if(i == M - sw - 1 )
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int j = sw + idy; j < N - sw; j += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + 1;
          int index = (N-2*sw)*(k-sw) + (j-sw);
          ap[temp1]+=A[temp1];
          //if(k == K - sw - 1 )
            buffer[index] = ap[temp1];
        }}
      } else {
        //if(i < M - sw - 1)
        for (int k = sw + idx; k < K - sw; k += blockDim.x*gridDim.x) {
        for (int j = sw + idy; j < N - sw; j += blockDim.y*gridDim.y) {
          int temp1 = N*M*k + M*j + i;
          int temp2 = temp1 + 1;
          ap[temp1]+=A[temp1];
          if(i < M - sw - 1)
            ap[temp2] += ap[temp1];
        }}
      }

    }

    template<typename T>
    void ker_buffer_csum_x_const(T *ap, T *A, int *host_values, int sw, int size, T *buffer, int type) {

      const int xs = host_values[0]; const int xe = host_values[1];
      const int ys = host_values[2]; const int ye = host_values[3];
      const int zs = host_values[4]; const int ze = host_values[5];
      
      const int M = xe - xs;
      const int N = ye - ys;
      const int K = ze - zs;

      const unsigned int dimBlock0 = ze;
      const unsigned int dimBlock1 = ye;
      
      const unsigned int size_of_values = sizeof(int)*6;
      int* dev_values;
      cudaMalloc((void**)&dev_values, size_of_values);
      cudaMemcpy(dev_values, host_values, size_of_values, cudaMemcpyHostToDevice);

      dim3 dimBlock(dimBlock0, dimBlock1, 1);
      dim3 dimThread(threadsPerBlock, 1, 1);

      ker1_buffer_csum_x_const<<<dimBlock, dimThread>>>(ap, A, buffer, dev_values, sw, size, type);
      cudaDeviceSynchronize();


      dim3 dimBlock2(dimBlock0, dimBlock1, 1);
      dim3 dimThread2(1, 1, 1);
      for (int x = sw; x < M - sw; x++) {
        ker2_buffer_csum_x_const<<<dimBlock2, dimThread2>>>(ap, A, buffer, dev_values, sw, size, type, x);
        cudaDeviceSynchronize();
      }

      cudaFree(dev_values);

    }

    template void ker_buffer_csum_x_const<int>(
        int *ap, int *A, int *host_values, 
        int sw, int size, int *buffer, int type);
    template void ker_buffer_csum_x_const<float>(
        float *ap, float *A, int *host_values, 
        int sw, int size, float *buffer, int type);
    template void ker_buffer_csum_x_const<double>(
        double *ap, double *A, int *host_values, 
        int sw, int size, double *buffer, int type);



}}}

#endif
