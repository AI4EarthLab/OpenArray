#include "CUDA.hpp"

#ifdef __HAVE_CUDA__
#include <cmath>
namespace oa{
    namespace gpu{

    void initialize_gpu(MPI_Comm comm){
        int rank = oa::MPI::global()->rank(comm);
        int size = oa::MPI::global()->size(comm);
        int nDevices;
        CUDA_CHECK(cudaGetDeviceCount(&nDevices));
        if(nDevices < rank){
             std::cerr << "[OpenArray] Not enough GPUs found, require " <<size<<"GPUs, found "<<nDevices<<" GPUs\n";
             throw std::runtime_error("Not enough GPUs found but OpenArray compiled with CUDA support.");
        }
        CUDA_CHECK(cudaSetDevice(rank));
    }

    std::pair<int, int> SizeToBlockThreadPair(int n)
    {
        assert(n);
        int logn;
        asm("\tbsr %1, %0\n"
            : "=r"(logn)
            : "r"(n - 1));
        logn = logn > 9 ? 9 : (logn < 4 ? 4 : logn);
        ++logn;
        int threads = 1 << logn;
        int blocks = (n + threads - 1) >> logn;
        blocks = blocks > 65535 ? 65535 : blocks;
        return std::make_pair(blocks, threads);
    }
 }
} // namespace oa
#endif
