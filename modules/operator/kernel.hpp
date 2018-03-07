
#ifndef __OP_KERNEL_HPP__
#define __OP_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "internal.hpp"
#include "../../Grid.hpp"
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
    ///:set type = k[0]
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

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ${name}$...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      int3 lbound = {{0, 0, 0}};
      int3 rbound = {{0, 0, 0}};
      ///:if name[1:] == "xb"
      lbound = {{1, 0, 0}};
      ///:elif name[1:] == "xf"
      rbound = {{1, 0, 0}};
      ///:elif name[1:] == "yb"
      lbound = {{0, 1, 0}};
      ///:elif name[1:] == "yf"
      rbound = {{0, 1, 0}};
      ///:elif name[1:] == "zb"
      lbound = {{0, 0, 1}};
      ///:elif name[1:] == "zf"
      rbound = {{0, 0, 1}};
      ///:elif name[1:] == 'xc'
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};
      ///:elif name[1:] == 'yc'
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};
      ///:elif name[1:] == 'zc'      
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};
      ///:endif

      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), ${type}$);
      // printf("pos:%d, ${type}$, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"${grid}$"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"${grid}$"<<endl;
        grid_dt = gridptr->get_data_type();
        grid_buffer = gridptr->get_buffer();
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      ///:mute
      ///:if name[1].lower() == 'x'
      ///:set di = 0
      ///:elif name[1].lower() == 'y'
      ///:set di = 1
      ///:elif name[1].lower() == 'z'
      ///:set di = 2
      ///:endif
      ///:endmute
      int cnt = 0;
      switch(grid_dt) {
        ///:for t in ['int','float','double']
      case DATA_${t.upper()}$:
          oa::funcs::update_ghost_start(u, reqs, ${di}$);
          oa::internal::${name}$_${grid}$_calc_inside<T1, T2, ${t}$>(ans,
                  buffer, (${t}$*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::${name}$_${grid}$_calc_outside<T1, T2, ${t}$>(ans,
                  buffer, (${t}$*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
          ///:endfor
      }

      return ap;
    }

    ///:endfor
    ///:endfor

  }
}
#endif
