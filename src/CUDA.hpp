#ifndef OA_CUDA_HPP__
#define OA_CUDA_HPP__

#ifdef __HAVE_CUDA__
#include <stdexcept>
#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "MPI.hpp"
#include "except.hpp"
#include <iostream>

#define CUDA_CHECK(stmt) do {                              \
    cudaError_t err = stmt;                                \
    if (err != cudaSuccess) {                              \
      std::cerr << "CUDA failure in " << #stmt << std::endl\
                << cudaGetErrorString(err) << std::endl;   \
      throw oa::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

#define CUBLAS_CHECK(stmt) do {                            \
    cublasStatus_t stat = stmt;                            \
    if (stat != CUBLAS_STATUS_SUCCESS) {                   \
      std::cerr << "CUBLAS failure in " << #stmt           \
                << std::endl << stat << std::endl;         \
      throw oa::cuda_exception(#stmt);                    \
    }                                                      \
  } while(0)

namespace oa{
  namespace gpu{
    std::pair<int, int> SizeToBlockThreadPair(int n);
    void initialize_gpu(MPI_Comm comm);

  }
} // namespace oa
#endif
#endif
