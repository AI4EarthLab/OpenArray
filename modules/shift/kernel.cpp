
#include "../../Function.hpp"
#include "../../ArrayPool.hpp"

namespace oa{
  namespace kernel{
    ArrayPtr kernel_shift(vector<ArrayPtr> &ops_ap){
      ArrayPtr ap = ops_ap.at(0);
      ArrayPtr direction = ops_ap.at(1);
      Shape s = ap->shape();

      int* id = (int*)direction->get_buffer();
      Box b(0,s[0], 0, s[1], 0, s[2]);

      Box dst_b(id[0], id[0]+s[0],
              id[1],id[1]+s[1],
              id[2],id[2]+s[2]);

      Box src_b(-id[0], -id[0]+s[0],
              -id[1],-id[1]+s[1],
              -id[2],-id[2]+s[2]);

      // printf("=====================\n");
      // b.display("b = ");
      // dst_b.display("dst_b = ");
      // src_b.display("src_b = ");
      
      Box dst_b1 = b.get_intersection(dst_b);
      // dst_b1.display("dst_b1 = ");      
      Box src_b1 = b.get_intersection(src_b);
      // src_b1.display("src_b1 = ");
      // printf("=====================\n");
      
      //duplicate an arrray
      ArrayPtr dst_ap =
        ArrayPool::global()->get(
            ap->get_partition(),
            ap->get_data_type());
      dst_ap->set_zeros();
      oa::funcs::set(dst_ap, dst_b1, ap, src_b1);
      //dst_ap->display("dst_ap = ");
      
      return dst_ap;
    }
  }
}
