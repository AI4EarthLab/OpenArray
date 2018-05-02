/*
 * Kernel.hpp
 * kernel function declarations
 *
=======================================================*/

#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

#include "NodePool.hpp"
#include "NodeDesc.hpp"
#include "Function.hpp"
#include "Internal.hpp"
#include <vector>
using namespace std;

///:mute
///:include "NodeType.fypp"
///:endmute
///:for m in MODULES
#include "modules/${m}$/kernel.hpp"
///:endfor

namespace oa {
  namespace kernel {

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] in ['G','']]
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // return ANS = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);

    ///:endfor

  }
}
#endif
