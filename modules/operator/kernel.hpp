
#ifndef __OP_KERNEL_HPP__
#define __OP_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "internal.hpp"
#include <vector>
using namespace std;

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa {
  namespace kernel {

    ///:for k in [i for i in L if i[3] == 'D']
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // return ANS = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);
    ///:endfor

    ///:for k in [i for i in L if i[3] == 'D']
    ///:set name = k[1]
    // crate kernel_${name}$
    // A = ${name}$(U)
    template<typename T>
    void t_kernel_${name}$(ArrayPtr& ap, ArrayPtr& u) {
      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      int3 lbound = {0, 0, 0};
      int3 rbound = {0, 0, 0};
      ///:if name[1:] == "xb"
      lbound = {1, 0, 0};
      ///:elif name[1:] == "xf"
      rbound = {1, 0, 0};
      ///:elif name[1:] == "yb"
      lbound = {0, 1, 0};
      ///:elif name[1:] == "yf"
      rbound = {0, 1, 0};
      ///:elif name[1:] == "zb"
      lbound = {0, 0, 1};
      ///:elif name[1:] == "zf"
      rbound = {0, 0, 1};
      ///:elif name[1:] == 'xc'
      lbound = {1, 0, 0};
      rbound = {1, 0, 0};
      ///:elif name[1:] == 'yc'
      lbound = {0, 1, 0};
      rbound = {0, 1, 0};
      ///:elif name[1:] == 'zc'      
      lbound = {0, 0, 1};
      rbound = {0, 0, 1};
      ///:endif

      double* ans = (double*) ap->get_buffer();
      T* buffer = (T*) u->get_buffer();

      /*
        to chose the right bind grid
      */
        

      vector<MPI_Request> reqs;
      oa::funcs::update_ghost_start(u, reqs, -1);
      oa::internal::${name}$_calc_inside<T>(ans,
              buffer, lbound, rbound, sw, sp, S);
      oa::funcs::update_ghost_end(reqs);
      oa::internal::${name}$_calc_outside<T>(ans,
              buffer, lbound, rbound, sw, sp, S);

    }

    ///:endfor

  }
}
#endif
