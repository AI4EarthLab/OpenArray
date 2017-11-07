#include "Kernel.hpp"
#include "ArrayPool.hpp"
#include "utils/utils.hpp"
#include "Internal.hpp"
#include <functional>

namespace oa {
  namespace kernel {
    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if i[3] == 'A']
    ///:set name = k[1]
    ///:set sy = k[2]
    // crate kernel_${name}$
    // A = U ${sy}$ V
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
	///:mute
	///:set i = 0
	///:include "kernel_type.fypp"
	///:endmute
	//create kernel_table
	///:for i in T
	///:set id = i[0]
	///:set type1 = i[1]
	///:set type2 = i[2]
	///:set type3 = i[3]
	      kernel_table[${id}$] = t_kernel_${name}$<${type1}$, ${type2}$, ${type3}$>;
	///:endfor
	      has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

  ///:endfor
		
  }
}
