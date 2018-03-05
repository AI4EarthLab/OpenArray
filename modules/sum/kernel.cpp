
#include "kernel.hpp"

namespace oa{
  namespace kernel{
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
      
      Box box_x(s[0] - 1  , s[0]  , 0         , s[1]  , 0         , s[2]  );
      Box box_y(0         , s[0]  , s[1] - 1  , s[1]  , 0         , s[2]  );
      Box box_z(0         , s[0]  , 0         , s[1]  , s[2] - 1  , s[2]  );
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
