#ifndef __GPU_KERNEL_HPP__
#define __GPU_KERNEL_HPP__
#ifdef __HAVE_CUDA__
#include "CUDA.hpp"
#include "Partition.hpp"
#include "Array.hpp"
namespace  oa {
 namespace gpu{
   ArrayPtr l2g_gpu(ArrayPtr& lap);


   template<class T>
   extern void set_gpu(ArrayPtr& A, const Box& ref_box,
            T* buf, Shape& buf_shape);
} //namespace gpu
} // namespace oa

#endif
#endif
