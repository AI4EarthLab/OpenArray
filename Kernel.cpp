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
    
    ///:mute
    ///:set K = [['gt','>'], ['ge', '>='], ['lt', '<'],['le', '<='], &
                 ['eq','=='], ['ne','/='],['and','&&'],['or','||']]
    ///:endmute
    ///:for t in K
    ///:set name = t[0]
    ///:set sy = t[1]
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
        kernel_table[${id}$] = t_kernel_${name}$<int, ${type2}$, ${type3}$>;
        ///:endfor
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    ///:endfor

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if (i[3] == 'C' and i[2] != '+' and i[2] != '-')]
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // crate kernel_${name}$
    // A = ${ef}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int dt = DATA_DOUBLE;

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      
      switch(u_dt) {
        case DATA_INT:
          oa::internal::buffer_${name}$(
            (double *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size()
          );
          break;
        case DATA_FLOAT:
          oa::internal::buffer_${name}$(
            (double *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size()
          );
          break;
        case DATA_DOUBLE:
          oa::internal::buffer_${name}$(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size()
          );
          break;
      }
      return ap;
    }

    ///:endfor

    ///:for k in [['uplus','+'], ['uminus','-']]
    ///:set name = k[0]
    ///:set sy = k[1]
    // crate kernel_${name}$
    // A = ${sy}$(A)
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int dt = u_dt;

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      
      switch(u_dt) {
        case DATA_INT:
          oa::internal::buffer_${name}$(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size()
          );
          break;
        case DATA_FLOAT:
          oa::internal::buffer_${name}$(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size()
          );
          break;
        case DATA_DOUBLE:
          oa::internal::buffer_${name}$(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size()
          );
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

    ///:for k in ['min', 'max']
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

    // crate kernel_csum
    // A = U > V
    ArrayPtr kernel_csum(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap; 
      ArrayPtr v = ops_ap[1];
      int type = ((int*)v->get_buffer())[0];

      int u_dt = u->get_data_type();

      switch(type){
        case 0://sum to scalar
          switch(u_dt) {
            case DATA_INT:
              ap = t_kernel_sum_scalar<int>(ops_ap);
              break;
            case DATA_FLOAT:
              ap = t_kernel_sum_scalar<float>(ops_ap);
              break;
            case DATA_DOUBLE:
              ap = t_kernel_sum_scalar<double>(ops_ap);
              break;
            default:
              std::cout<<"error"<<std::endl;
              break;
          }
          break;
        case 1://csum to x
          switch(u_dt) {
            case DATA_INT:
              ap = t_kernel_csum_x<int>(ops_ap);
              break;
            case DATA_FLOAT:
              ap = t_kernel_csum_x<float>(ops_ap);
              break;
            case DATA_DOUBLE:
              ap = t_kernel_csum_x<double>(ops_ap);
              break;
            default:
              std::cout<<"error"<<std::endl;
              break;
          }
          break;
        case 2://csum to y
          switch(u_dt) {
            case DATA_INT:
              ap = t_kernel_csum_y<int>(ops_ap);
              break;
            case DATA_FLOAT:
              ap = t_kernel_csum_y<float>(ops_ap);
              break;
            case DATA_DOUBLE:
              ap = t_kernel_csum_y<double>(ops_ap);
              break;
            default:
              std::cout<<"error"<<std::endl;
              break;
          }
          break;
        case 3://csum to z
          switch(u_dt) {
            case DATA_INT:
              ap = t_kernel_csum_z<int>(ops_ap);
              break;
            case DATA_FLOAT:
              ap = t_kernel_csum_z<float>(ops_ap);
              break;
            case DATA_DOUBLE:
              ap = t_kernel_csum_z<double>(ops_ap);
              break;
            default:
              std::cout<<"error"<<std::endl;
              break;
          }
          break;
        default:
          std::cout<<"error"<<std::endl;
          break;
      }
      return ap;
    }

    ArrayPtr kernel_sum(vector<ArrayPtr> &ops_ap) {
      ArrayPtr ap; 
      ap = kernel_csum(ops_ap);
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      int type = ((int*)v->get_buffer())[0];
      ArrayPtr sub_ap;
      int xs, xe, ys, ye, zs, ze, sw;
      u->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
      Shape s = u->shape();
      
      Box box_x(0, 0, 0, s[1]-1, 0, s[2]-1);
      Box box_y(0, s[0]-1, 0, 0, 0, s[2]-1);
      Box box_z(0, s[0]-1, 0, s[1]-1, 0, 0);
      switch(type){
        case 0:
          return ap;
          break;
        case 1:
          {
            sub_ap = oa::funcs::subarray(ap, box_x);
            break;
          }
        case 2:
          {
            sub_ap = oa::funcs::subarray(ap, box_y);
            break;
          }
        case 3:
          {
            sub_ap = oa::funcs::subarray(ap, box_z);
            break;
          }
        default:
          std::cout<<"error"<<std::endl;
          break;

      }

      return sub_ap;
    }
  }
}
