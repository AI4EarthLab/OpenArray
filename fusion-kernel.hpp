#ifndef __FUSION_KERNEL_HPP__
#define __FUSION_KERNEL_HPP__

#include "common.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"

#:set kernel_file = "fusion-kernels"
#:if os.path.isfile(kernel_file)
#:set lines = io.open(kernel_file).read().split('\n')
#:for i in lines[:-1]
#:set line = i.split(' ')
#:set key = line[0]
#:set expr = line[1]
ArrayPtr kernel_${key}$(vector<ArrayPtr> &ops, int dt, int size);
#:endfor
#:endif

#endif