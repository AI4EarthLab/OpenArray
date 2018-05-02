
#include "kernel.hpp"
#include "../../Array.hpp"
#include "internal.hpp"
#include "../../Grid.hpp"
#include <bitset>

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa {
  namespace kernel {
    ///:for k in [i for i in L if i[3] == 'D']
    ///:set type = k[0]
    ///:set name = k[1]
    // crate kernel_${name}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        ///:mute
        ///:include "kernel_type.fypp"
        ///:endmute
        //create kernel_table
        
        ///:for i in T_INT
        ///:set id = i[0]
        ///:set type_out = i[2]
        ///:set type_in = i[3]
        ///:set grid = i[4]
        kernel_table[0][${id}$] =
          t_kernel_${name}$_${grid}$<${type_out}$, ${type_in}$>;
        ///:endfor

        ///:for i in T_FLOAT
        ///:set id = i[0]
        ///:set type_out = i[2]
        ///:set type_in = i[3]
        ///:set grid = i[4]
        kernel_table[1][${id}$] =
          t_kernel_${name}$_${grid}$<${type_out}$, ${type_in}$>;
        ///:endfor

        ///:for i in T_DOUBLE
        ///:set id = i[0]
        ///:set type_out = i[2]
        ///:set type_in = i[3]
        ///:set grid = i[4]
        kernel_table[2][${id}$] =
          t_kernel_${name}$_${grid}$<${type_out}$, ${type_in}$>;
        ///:endfor

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, ${type}$)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform ${name.upper()}$ on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      ///:if name[1].lower()   == 'x'
      if(us[0] < 2) assert(false && err_msg.c_str());
      ///:elif name[1].lower() == 'y'
      if(us[1] < 2) assert(false && err_msg.c_str());
      ///:elif name[1].lower() == 'z'
      if(us[2] < 2) assert(false && err_msg.c_str());
      ///:endif
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    ///:endfor
  }
}
