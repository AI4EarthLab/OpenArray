
#include "../../Function.hpp"
#include "../../ArrayPool.hpp"

namespace oa{
  namespace kernel{
    ArrayPtr kernel_shift(vector<ArrayPtr> &ops_ap){
      ArrayPtr ap = ops_ap.at(0);

      ArrayPtr direction = ops_ap.at(1);
      Shape s = ap->shape();

      #ifdef __HAVE_CUDA__
      direction->memcopy_gpu_to_cpu();
      int* id = (int*)direction->get_cpu_buffer();
      #else
      int* id = (int*)direction->get_buffer();
      #endif
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


    ArrayPtr kernel_circshift(vector<ArrayPtr> &ops_ap){
      ArrayPtr ap = ops_ap.at(0);

      ArrayPtr direction = ops_ap.at(1);
      Shape s = ap->shape();
      Box b(0,s[0], 0, s[1], 0, s[2]);

      #ifdef __HAVE_CUDA__
      direction->memcopy_gpu_to_cpu();
      int* id = (int*)direction->get_cpu_buffer();
      #else
      int* id = (int*)direction->get_buffer();
      #endif

      id[0] = -id[0];
      id[1] = -id[1];
      id[2] = -id[2];
      
      int bx = (id[0] >= 0) ? id[0] : (s[0] + id[0]);
      int by = (id[1] >= 0) ? id[1] : (s[1] + id[1]);
      int bz = (id[2] >= 0) ? id[2] : (s[2] + id[2]);

      // printf("bx=%d, by=%d, bz=%d\n", bx, by, bz);
      
      int x[2][2] = {{0, bx}, {bx, s[0]}};
      int y[2][2] = {{0, by}, {by, s[1]}};
      int z[2][2] = {{0, bz}, {bz, s[2]}};

      int xs1, xe1, ys1, ye1, zs1, ze1;
      int xs2, xe2, ys2, ye2, zs2, ze2;

      //duplicate an arrray
      ArrayPtr dst_ap =
        ArrayPool::global()->get(
            ap->get_partition(),
            ap->get_data_type());
      
      for(int k = 0; k < 2; k++){
        for(int j = 0; j < 2; j++){
          for(int i = 0; i < 2; i++){
            xs1 = x[i][0]; xe1 = x[i][1];
            ys1 = y[j][0]; ye1 = y[j][1];            
            zs1 = z[k][0]; ze1 = z[k][1];

            // printf("xs1=%d, xe1=%d, ys1=%d, ye1=%d, zs1=%d, ze1=%d\n",
            //xs1, xe1, ys1, ye1, zs1, ze1);

            if(xe1 > xs1 && ye1 > ys1 && ze1 > zs1){

              Box src_b(xs1, xe1, ys1, ye1, zs1, ze1);

              if(xs1 == 0){
                xs2 = s[0] - (xe1 - xs1);
                xe2 = s[0];
              }else if(xe1 == s[0]){
                xs2 = 0;
                xe2 = xe1 - xs1;
              }

              if(ys1 == 0){
                ys2 = s[1] - (ye1 - ys1);
                ye2 = s[1];
              }else if(ye1 == s[1]){
                ys2 = 0;
                ye2 = ye1 - ys1;
              }

              if(zs1 == 0){
                zs2 = s[2] - (ze1 - zs1);
                ze2 = s[2];
              }else if(ze1 == s[2]){
                zs2 = 0;
                ze2 = ze1 - zs1;
              }

              Box dst_b(xs2, xe2, ys2, ye2, zs2, ze2);

              // src_b.display("src_b = ");
              // dst_b.display("dst_b = ");
              
              oa::funcs::set(dst_ap, dst_b, ap, src_b);
            }
          }
        }
      }
      
      //dst_ap->display("dst_ap = ");
      
      return dst_ap;
    }
  }
}
