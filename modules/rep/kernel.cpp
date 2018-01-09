
#include "kernel.hpp"
#include "../../MPI.hpp"

namespace oa{
  namespace kernel{
    
    ArrayPtr kernel_rep_with_partition(vector<ArrayPtr> &ops_ap,
            bool same_partition)
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
      // if(same_partition){
      //   printf("$$$$#: %d %d %d\n", x, y, z);        
      //   ap = ArrayPool::global()->get(
      //       A->get_partition(),
      //       A->get_data_type());
      //   ap->display("ap = ");
      //   printf("$$$$*: %d %d %d\n", x, y, z);
      // }else{
      ap = ArrayPool::global()->get(MPI::global()->comm(),
              {{s[0]*x, s[1]*y, s[2]*z}}, sw,
              A->get_data_type());
        //}
      int xs, xe, ys, ye, zs, ze;
      //std::cout<<"sw="<<sw<<std::endl;
      xs = ys = zs = 0;
      xe = s[0];
      ye = s[1];
      ze = s[2];
      for(int i = 0; i < x; i++){
        ys = 0;
        zs = 0;
        ye = s[1];
        ze = s[2];
        for(int j = 0; j < y; j++){
          zs = 0;
          ze = s[2];
          for(int k = 0; k < z; k++){
            Box box(xs, xe, ys, ye, zs, ze);
            oa::funcs::set(ap, box, A);
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

    ArrayPtr kernel_rep(vector<ArrayPtr> &ops_ap){
      return kernel_rep_with_partition(ops_ap, false);
    }

  }
}
