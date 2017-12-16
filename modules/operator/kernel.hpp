
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

    ///:mute
    ///:include "kernel_type.fypp"
    ///:endmute
    
    ///:for i in T_INT
    ///:set grid = i[4]
    template<typename T1, typename T2>
    ArrayPtr t_kernel_${name}$_${grid}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);

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

      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();

      /*
        to chose the right bind grid
      */
        

      vector<MPI_Request> reqs;
      oa::funcs::update_ghost_start(u, reqs, -1);
      oa::internal::${name}$_${grid}$_calc_inside<T1, T2>(ans,
              buffer, lbound, rbound, sw, sp, S);
      oa::funcs::update_ghost_end(reqs);
      oa::internal::${name}$_${grid}$_calc_outside<T1, T2>(ans,
              buffer, lbound, rbound, sw, sp, S);

      return ap;
    }

    ///:endfor
    ///:endfor

  }
}
#endif
