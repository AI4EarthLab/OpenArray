
#include "kernel.hpp"
///:include "../../NodeType.fypp"

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


    ///:for k in [i for i in L if i[3] == 'I']
    ///:set name = k[1]
    ///:set kernel_name = k[2]
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        ///:mute
        ///:set i = 0
        ///:include "../../kernel_type.fypp"
        ///:endmute
        ///:for i in T
        ///:set id = i[0]
        kernel_table[${id}$] =
          t_kernel_${name}$<${i[1]}$, ${i[2]}$, ${i[3]}$>;
        ///:endfor
        has_init = true;
      }

      const ArrayPtr& u = ops_ap[0];
      const ArrayPtr& v = ops_ap[1];
      
      const int u_dt = u->get_data_type();
      const int v_dt = u->get_data_type();
      const int r_dt = oa::utils::cast_data_type(u_dt, v_dt);
      int case_num = r_dt * 9 + u_dt * 3 + v_dt;

      
      ArrayPtr ap = kernel_table[case_num](ops_ap);
      return ap;
    }
    ///:endfor

  }
}
