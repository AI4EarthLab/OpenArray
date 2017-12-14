#include "Kernel.hpp"
#include "ArrayPool.hpp"
#include "utils/utils.hpp"
#include "Internal.hpp"
#include <functional>

namespace oa {
  namespace kernel {

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

    ///:for k in ['min', 'max', 'min_at', 'max_at', &
  'abs_max', 'abs_min', 'abs_max_at', 'abs_min_at']
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


    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
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


    ArrayPtr kernel_rep(vector<ArrayPtr> &ops_ap)
    {
      const ArrayPtr& A = ops_ap[0];
      const ArrayPtr& d = ops_ap[1];
    
      int* rep_dim = (int*)d->get_buffer();
      int x, y, z;  
      x = rep_dim[0]; y = rep_dim[1]; z = rep_dim[2];
      
      ArrayPtr ap;
      Shape s = A->shape();
      int sw = A->get_partition()->get_stencil_width();
      //std::cout<<"s0:2:"<<s[0]<<","<<s[1]<<","<<s[2]<<","<<std::endl;
      ap = oa::funcs::zeros(MPI_COMM_WORLD, {s[0]*x, s[1]*y, s[2]*z}, sw,
                            A->get_data_type());
      int xs, xe, ys, ye, zs, ze;
      //std::cout<<"sw="<<sw<<std::endl;
      xs = ys = zs = 0;
      xe = s[0] - 1;
      ye = s[1] - 1;
      ze = s[2] - 1;
      for(int i = 0; i < x; i++){
        ys = 0;
        zs = 0;
        ye = s[1] - 1;
        ze = s[2] - 1;
        for(int j = 0; j < y; j++){
          zs = 0;
          ze = s[2] - 1;
          for(int k = 0; k < z; k++){
            Box box(xs, xe, ys, ye, zs, ze);
            oa::funcs::set(ap, box, A);
            //box.display();
            zs += s[2];
            ze += s[2];
          }
          ys += s[1];
          ye += s[1];
        }
        xs += s[0];
        xe += s[0];
      }
      return ap;
    }
  }
}
