
#include "kernel.hpp"
#include "../../Array.hpp"
#include "kernel.hpp"
#include "internal.hpp"

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa{
  namespace kernel{
    
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
    

    ///:for t in [i for i in L if i[3] == 'B' or i[3] == 'F']
    ///:set name = t[1]
    ///:set sy = t[2]
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
        ///:include "../../kernel_type.fypp"
        ///:endmute
        //create kernel_table
        ///:for i in T
        ///:set id = i[0]
        ///:set type1 = i[1]
        ///:set type2 = i[2]
        ///:set type3 = i[3]
        kernel_table[${id}$] = t_kernel_${name}$<int, ${type2}$, ${type3}$>;
        ///:endfor
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    ///:endfor


    ///!:for k in [i for i in L if (i[3] == 'C')]
    ///:for k in [i for i in L if i[3] == 'C']
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // crate kernel_${name}$
    // A = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_${name}$(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_${name}$(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_${name}$(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    ///:endfor

    ArrayPtr kernel_pow(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = DATA_DOUBLE;

      ap = ArrayPool::global()->get(u->get_partition(), dt);

      // (u)**v, v must be a scalar
      assert(v->is_seqs_scalar());

      double m;
      switch(v_dt) {
      case DATA_INT:
        m = ((int*)v->get_buffer())[0];
        break;
      case DATA_FLOAT:
        m = ((float*)v->get_buffer())[0];
        break;
      case DATA_DOUBLE:
        m = ((double*)v->get_buffer())[0];
        break;
      }

      switch(u_dt) {
      case DATA_INT:
        oa::internal::buffer_pow(
            (double *) ap->get_buffer(),
            (int *) u->get_buffer(),
            m, ap->buffer_size()
                                 );
        break;
      case DATA_FLOAT:
        oa::internal::buffer_pow(
            (double *) ap->get_buffer(),
            (float *) u->get_buffer(),
            m, ap->buffer_size()
                                 );
        break;
      case DATA_DOUBLE:
        oa::internal::buffer_pow(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            m, ap->buffer_size()
                                 );
        break;
      }
      return ap;
    }


    ArrayPtr kernel_not(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int dt = DATA_INT;

      ap = ArrayPool::global()->get(u->get_partition(), dt);

      switch(u_dt) {
      case DATA_INT:
        oa::internal::buffer_not(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size()
                                 );
        break;
      case DATA_FLOAT:
        oa::internal::buffer_not(
            (int *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size()
                                 );
        break;
      case DATA_DOUBLE:
        oa::internal::buffer_not(
            (int *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size()
                                 );
        break;
      }
      return ap;
    }
  }
}
