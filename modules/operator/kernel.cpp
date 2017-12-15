
#include "kernel.hpp"
#include "../../Array.hpp"
#include "internal.hpp"

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa {
  namespace kernel {
    ///:for k in [i for i in L if i[3] == 'D']
    ///:set name = k[1]
    // crate kernel_${name}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int dt = DATA_DOUBLE;

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      switch(u_dt) {
      case DATA_INT:
        t_kernel_${name}$<int>(ap, u);
        break;
      case DATA_FLOAT:
        t_kernel_${name}$<float>(ap, u);
        break;
      case DATA_DOUBLE:
        t_kernel_${name}$<double>(ap, u);
        break;
      }

      return ap;
    }

    ///:endfor
  }
}
