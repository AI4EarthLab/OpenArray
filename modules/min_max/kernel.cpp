
#include "kernel.hpp"

namespace oa{
  namespace kernel{

    ///:for k in ['min', 'max', 'min_at', 'max_at',  'abs_max', 'abs_min', 'abs_max_at', 'abs_min_at']
    ///:set name = k
    // crate kernel_${name}$
    // A = ${name}$(A)
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_${name}$<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_${name}$<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_${name}$<double>(ops_ap);
        break;
      }
      return ap;
    }
    ///:endfor

  }
}
