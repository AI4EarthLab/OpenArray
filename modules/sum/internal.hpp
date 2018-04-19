
#ifndef __SUM_INTERNAL_HPP__
#define __SUM_INTERNAL_HPP__

namespace oa{
  namespace internal{

    template<typename T>
      void buffer_sum_scalar_const(T *val, T *A, Box box, int sw, int size) {
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;
        *val = 0;

        int cnt = 0;
        for (int k = zs; k < ze; k++) {
          for (int j = ys; j < ye; j++) {
            for (int i = xs; i < xe; i++) {
              if ((xs + sw <= i && i < xe - sw) &&
                  (ys + sw <= j && j < ye - sw) &&
                  (zs + sw <= k && k < ze - sw)) {
                *val += A[cnt];
              }
              cnt++;
            }
          }
        }
      }

    template<typename T>
      void buffer_csum_x_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        int cnt = 0;
        int dcnt = 0;
        if(type == 2) 
          for(int i = 0; i < (ye-ys-2*sw)*(ze-zs-2*sw); i++)
            buffer[i] = 0;

        int index = 0;
        for (int k = zs; k < ze; k++) {
          for (int j = ys; j < ye; j++) {
            for (int i = xs; i < xe; i++) {
              if ((xs + sw <= i && i < xe - sw) &&
                  (ys + sw <= j && j < ye - sw) &&
                  (zs + sw <= k && k < ze - sw)) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                if((i == xs + sw) && (type == 1 || type == 0))
                  ap[temp1] = buffer[index++];
                else 
                  ap[temp1] = 0;
              }
            }
          }
        }

        index = 0;
        for (int k = zs; k < ze; k++) {
          for (int j = ys; j < ye; j++) {
            for (int i = xs; i < xe; i++) {
              if ((xs + sw <= i && i < xe - sw) &&
                  (ys + sw <= j && j < ye - sw) &&
                  (zs + sw <= k && k < ze - sw)) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                int temp2 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs+1);
                ap[temp1]+=A[temp1];
                if(i < xe - sw - 1){
                  ap[temp2] += ap[temp1];
                }
                if((i == xe - sw - 1) && (type == 1 || type == 2))
                  buffer[index++] = ap[temp1];
              }
            }
          }
        }
      }


    template<typename T>
      void buffer_csum_y_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        int cnt = 0;
        int dcnt = 0;
        if(type == 2) 
          for(int i = 0; i < (xe-xs-2*sw)*(ze-zs-2*sw); i++)
            buffer[i] = 0;

        int index = 0;
        for (int k = zs; k < ze; k++) {
          for (int j = ys; j < ye; j++) {
            for (int i = xs; i < xe; i++) {
              if ((xs + sw <= i && i < xe - sw) &&
                  (ys + sw <= j && j < ye - sw) &&
                  (zs + sw <= k && k < ze - sw)) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                if((j == ys + sw) && (type == 1 || type == 0))
                  ap[temp1] = buffer[index++];
                else 
                  ap[temp1] = 0;
              }
            }
          }
        }
        index = 0;
        for (int k = zs; k < ze; k++) {
          for (int i = xs; i < xe; i++) {
            for (int j = ys; j < ye; j++) {
              if ((xs + sw <= i && i < xe - sw) &&
                  (ys + sw <= j && j < ye - sw) &&
                  (zs + sw <= k && k < ze - sw)) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                int temp2 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j+1-ys)+(i-xs);
                ap[temp1]+=A[temp1];
                if(j < ye - sw -1){
                  ap[temp2] += ap[temp1];
                }
                if((j == ye - sw -1) && (type == 1 || type == 2))
                  buffer[index++] = ap[temp1];
              }
            }
          }
        }
      }

    template<typename T>
      void buffer_csum_z_const(T *ap, T *A, Box box, int sw, int size, T *buffer, int type) {
        //type:   top 2  mid 1  bottom 0
        int x = 0, y = 0, z = 0;
        int xs, xe, ys, ye, zs, ze;
        box.get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);

        int M = xe - xs;
        int N = ye - ys;
        int K = ze - zs;

        int cnt = 0;
        int dcnt = 0;
        if(type == 2) 
          for(int i = 0; i < (ye-ys-2*sw)*(xe-xs-2*sw); i++)
            buffer[i] = 0;

        int index = 0;
        for (int k = zs + sw; k < ze - sw; k++) {
          for (int j = ys + sw; j < ye - sw; j++) {
            for (int i = xs + sw; i < xe - sw; i++) {
              int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
              if((k == zs + sw) && (type == 1 || type == 0))
                ap[temp1] = buffer[index++];
              else 
                ap[temp1] = 0;
            }
          }
        }
        if(type == 1 || type == 2){
          index = 0;
          for (int k = zs + sw; k < ze - sw; k++) {
            for (int j = ys + sw; j < ye - sw; j++) {
              //#pragma simd
              for (int i = xs + sw; i < xe - sw; i++) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                int temp2 = (xe-xs)*(ye-ys)*(k+1-zs)+(xe-xs)*(j-ys)+(i-xs);
                ap[temp1]+=A[temp1];
                if(k < ze - sw - 1){
                  ap[temp2] += ap[temp1];
                }
                if(k == ze - sw - 1 )
                  buffer[index++] = ap[temp1];
                //buffer[(j-ys-sw)*(xe-xs-2*sw)+(i-xs-sw)] = ap[temp1];
              }
            }
          }
        }else{
          for (int k = zs + sw; k < ze - sw; k++) {
            for (int j = ys + sw; j < ye - sw; j++) {
#pragma simd
              for (int i = xs + sw; i < xe - sw; i++) {
                int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
                int temp2 = (xe-xs)*(ye-ys)*(k+1-zs)+(xe-xs)*(j-ys)+(i-xs);
                ap[temp1]+=A[temp1];
                if(k < ze - sw - 1){
                  ap[temp2] += ap[temp1];
                }
              }
            }
          }
        }
      }

  }
}

#endif
