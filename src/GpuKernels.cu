#ifdef  __HAVE_CUDA__
#include "CUDA.hpp"
#include <vector>
#include "GpuKernels.hpp"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "MPI.hpp"
#include "Array.hpp"
#include "common.hpp"
#include "Function.hpp"
namespace  oa {
 namespace gpu{

   template<typename T> 
   __global__ void ker_l2g_loop(T* gbuff, T* lbuff, int xs, int xe, int ys, int ye, int zs, int ze, int k, 
                                int lxs, int lxe, int lys, int lye,  int sw, 
                                thrust::device_ptr<int> clx, thrust::device_ptr<int> cly, 
                                thrust::device_ptr<int> clz, thrust::device_ptr<int>location)
   {
      int i = blockIdx.x * blockDim.x + threadIdx.x + xs;
      int j = blockIdx.y * blockDim.y + threadIdx.y + ys;
      if(i>=xe || j>=ye) return;
      int lk = k + clx[location[0]];
      int lj = j + cly[location[1]];
      int li = i + clz[location[2]];
      int gindex = (xe-xs+2*sw)*(ye-ys+2*sw)*(k-zs+sw)+(xe-xs+2*sw)*(j-ys+sw)+(i-xs+sw);
      int lindex = (lxe-lxs+2*sw)*(lye-lys+2*sw)*(k+sw)+(lxe-lxs+2*sw)*(j+sw)+(i+sw);
      gbuff[gindex] = lbuff[lindex];
      //__syncthreads();
   }


   ArrayPtr l2g_gpu(ArrayPtr& lap)
    {
      ArrayPtr gap;

      PartitionPtr lpp = lap->get_partition();

      Shape lsp = lpp->shape();
      Shape gsp = lsp;
      int sw = lpp->get_stencil_width(); 
      int datatype = lap->get_data_type();
      gap = oa::funcs::zeros(oa::MPI::global()->comm(), gsp, sw, datatype);
      PartitionPtr gpp = gap->get_partition();
      Box gbox = gpp->get_local_box();
      int xs, ys, zs, xe, ye, ze;
      gbox.get_corners(xs, xe, ys, ye, zs, ze);

      int lxs, lys, lzs, lxe, lye, lze;
      Box lbox =lpp->get_local_box();
      lbox.get_corners(lxs, lxe, lys, lye, lzs, lze);

      std::vector<int> clx = gpp->m_clx;
      std::vector<int> cly = gpp->m_cly;
      std::vector<int> clz = gpp->m_clz;

      int mpisize = MPI_SIZE;
      int myrank  = MPI_RANK;

      vector<int> location = gpp->get_procs_3d(myrank);
      thrust::device_vector<int> d_location(location);
     thrust::device_vector<int> d_clx(clx);
     thrust::device_vector<int> d_cly(cly);
     thrust::device_vector<int> d_clz(clz); 
     thrust::device_ptr<int> d_loc_ptr = d_location.data();
     thrust::device_ptr<int> d_clx_ptr = d_clx.data();
     thrust::device_ptr<int> d_cly_ptr = d_cly.data();
     thrust::device_ptr<int> d_clz_ptr = d_clz.data();
     
      dim3 threads_per_block(16,16);
     dim3 num_blocks((xe-xs+15)/16,(ye-ys+15)/16);
     switch(datatype) {
        case DATA_INT:
          {
            int *gbuff = (int *) gap->get_buffer();
            int *lbuff = (int *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
                ker_l2g_loop<<<num_blocks, threads_per_block>>>(gbuff, lbuff, xs, xe, ys, ye, zs, ze, k, 
                             lxs, lxe, lys, lye, sw, 
                            d_clx_ptr, d_cly_ptr, d_clz_ptr, d_loc_ptr);
            }
            break;
          }
        case DATA_FLOAT:
          {
            float *gbuff = (float *) gap->get_buffer();
            float *lbuff = (float *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
                ker_l2g_loop<<<num_blocks, threads_per_block>>>(gbuff, lbuff, xs, xe, ys, ye, zs, ze, k, 
                             lxs, lxe, lys, lye, sw, 
                            d_clx_ptr, d_cly_ptr, d_clz_ptr, d_loc_ptr);
            }
            break;
          }
        case DATA_DOUBLE:
          {
            double *gbuff = (double *) gap->get_buffer();
            double *lbuff = (double *) lap->get_buffer();
            for (int k = zs; k < ze; k++) {
                ker_l2g_loop<<<num_blocks, threads_per_block>>>(gbuff, lbuff, xs, xe, ys, ye, zs, ze, k, 
                             lxs, lxe, lys, lye, sw, 
                            d_clx_ptr, d_cly_ptr, d_clz_ptr, d_loc_ptr);
            }
            break;
          }
        default:
          std::cout<<"err"<<std::endl;
          break;
      }

      return gap;
    }

    template<typename T1, typename T2>
    __global__ void ker_set_gpu(T1* dst_buf, T2* buf, int x1, int y1, int z1, int x2, int y2, int z2, 
                        int bs0, int bs1, int buf_shape0, int buf_shape1, int M, int N, int k, int sw)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      int j = blockIdx.y * blockDim.y + threadIdx.y;
      if(i>=M || j>=N) return;
      int idx1 = (i + sw + x1) +
      (j + sw + y1) * bs0 +
      (k + sw + z1) * bs0 * bs1;
      int idx2 = (i + x2) +
      (j + y2) * buf_shape0 +
      (k + z2) * buf_shape0 * buf_shape1;
      dst_buf[idx1] = buf[idx2];
      //__syncthreads();
    }

    template<class T>
    void set_gpu(ArrayPtr& A, const Box& ref_box,
            T* buf, Shape& buf_shape){

      int num = buf_shape[0] * buf_shape[1] * buf_shape[2];
      assert(ref_box.shape() == buf_shape);
      
      Box local_box = A->get_local_box();
      Box local_ref_box = local_box.get_intersection(ref_box);

      oa_int3 offset_local =
        local_ref_box.starts() - local_box.starts();
      int x1 = offset_local[0];
      int y1 = offset_local[1];
      int z1 = offset_local[2];
      
      oa_int3 offset_ref =
        local_ref_box.starts() - ref_box.starts();
      int x2 = offset_ref[0];
      int y2 = offset_ref[1];
      int z2 = offset_ref[2];
      
      Shape slr = local_ref_box.shape();
      int M = slr[0];
      int N = slr[1];
      int P = slr[2];

      Shape bs = A->buffer_shape();

      const int sw = A->get_partition()->get_stencil_width();

      void* dst_buf = A->get_buffer();
      
      dim3 threads_per_block(16,16);
     dim3 num_blocks((M+15)/16,(N+15)/16);
      switch(A->get_data_type()){
      case (DATA_INT):
        for(int k = 0; k < P; ++k){
          ker_set_gpu<<<num_blocks, threads_per_block>>>( (int*)dst_buf, buf, x1, y1, z1, x2, y2, z2, 
                        bs[0], bs[1], buf_shape[0], buf_shape[1], M, N, k, sw);
        } 
        break;
      case (DATA_FLOAT):
        for(int k = 0; k < P; ++k){
          ker_set_gpu<<<num_blocks, threads_per_block>>>( (float*)dst_buf, buf, x1, y1, z1, x2, y2, z2, 
                        bs[0], bs[1], buf_shape[0], buf_shape[1], M, N, k, sw);
        } 
        break;
      case (DATA_DOUBLE):
        for(int k = 0; k < P; ++k){
          ker_set_gpu<<<num_blocks, threads_per_block>>>( (double*)dst_buf, buf, x1, y1, z1, x2, y2, z2, 
                        bs[0], bs[1], buf_shape[0], buf_shape[1], M, N, k, sw);
        } 
        break;
      }
    }

    template void set_gpu(ArrayPtr& A, const Box& ref_box,  double* buf, Shape& buf_shape);
    template void set_gpu(ArrayPtr& A, const Box& ref_box,  float* buf, Shape& buf_shape);
    template void set_gpu(ArrayPtr& A, const Box& ref_box,  int* buf, Shape& buf_shape);
} //namespace gpu
} // namespace oa

#endif
