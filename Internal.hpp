#ifndef __INTERNAL_HPP__
#define __INTERNAL_HPP__

#include <random>
#include "common.hpp"
#include "Box.hpp"
#include "utils/utils.hpp"
#include <vector>

extern "C"{
  void tic(const char* s);
  void toc(const char* s);
  void show_time(const char* s);
  void show_all();
}

namespace oa {
  namespace internal {
    template <typename T>
    void set_buffer_consts(T *buffer, int size, T val) {
      for (int i = 0; i < size; i++) buffer[i] = val;
    }

    //sub_A = sub(A, box)
    template <typename T>
    void get_buffer_subarray(T *sub_buffer, T *buffer, const Box &sub_box,
      const Box &box, int sw) {
      
      Shape sp = box.shape(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
			
      Box bd_box = box.boundary_box(sw);
      Box ref_box = sub_box.ref_box(bd_box);
      int xs, xe, ys, ye, zs, ze;
      ref_box.get_corners(xs, xe, ys, ye, zs, ze, sw);
      
      //ref_box.display("ref_box");

      int cnt = 0;
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            sub_buffer[cnt++] = buffer[k * M * N + j * M + i];
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    // set sub(A) = B
    template<typename T1, typename T2>
    void set_buffer_subarray(T1* buffer, T2* sub_buffer, const Box &box,
      const Box &sub_box, int sw) {

      Shape sp = box.shape(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
      
      Box bd_box = box.boundary_box(sw);
      Box ref_box = sub_box.ref_box(bd_box);
      int xs, xe, ys, ye, zs, ze;
      ref_box.get_corners(xs, xe, ys, ye, zs, ze, sw);
      
      //ref_box.display("ref_box");

      int cnt = 0;
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if (zs + sw <= k && k < ze - sw &&
                ys + sw <= j && j < ye - sw &&
                xs + sw <= i && i < xe - sw) {
              buffer[k * M * N + j * M + i] = sub_buffer[cnt];
            }
            cnt++;
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    // set sub(A) = const
    template<typename T1, typename T2>
    void set_buffer_subarray_const(T1* buffer, T2 val, const Box &box, 
      const Box &sub_box, int sw) {

      Shape sp = box.shape(sw);
      int M = sp[0];
      int N = sp[1];
      int P = sp[2];
      
      Box bd_box = box.boundary_box(sw);
      Box ref_box = sub_box.ref_box(bd_box);
      int xs, xe, ys, ye, zs, ze;
      ref_box.get_corners(xs, xe, ys, ye, zs, ze, sw);
      
      //ref_box.display("ref_box");

      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if (zs + sw <= k && k < ze - sw &&
                ys + sw <= j && j < ye - sw &&
                xs + sw <= i && i < xe - sw) {
              buffer[k * M * N + j * M + i] = val;
            }
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }



    template<typename T>
    void set_buffer_rand(T *buffer, int size) {
      srand(SEED);
      for (int i = 0; i < size; i++) {
        float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        buffer[i] = r;
      }
    }

    template<typename T>
    void set_buffer_seqs(T *buffer, const Shape& s, Box box, int sw) {
      int cnt = 0;
      int xs, xe, ys, ye, zs, ze;
      int M = s[0];
      int N = s[1];
      int P = s[2];
      //cout<<M<<" "<<N<<" "<<P<<endl;
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);
      //printf("%d %d %d %d %d %d\n", xs, xe, ys, ye, zs, ze);
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            buffer[cnt++] = k * M * N + j * M + i;
            //cout<<buffer[cnt-1]<<" ";
          }
          //cout<<endl;
        }
        //cout<<endl;
      }
    }

    template <typename T>
    void set_ghost_consts(T *buffer, const Shape &sp, T val, int sw = 1) {
      int M = sp[0] + 2 * sw;
      int N = sp[1] + 2 * sw;
      int P = sp[2] + 2 * sw;

      int cnt = 0;
      for (int k = 0; k < P; k++) {
        for (int j = 0; j < N; j++) {
          for (int i = 0; i < M; i++) {
            if ((sw <= k && k < P - sw) &&
                (sw <= j && j < N - sw) &&
                (sw <= i && i < M - sw)) {
              cnt++;
              continue;
            }
            buffer[cnt++] = val;
          }
        }
      }
    }

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if (i[3] == 'A' or i[3] == 'B' or i[3] == 'F')]
    ///:set name = k[1]
    ///:set sy = k[2]
    // A = B ${sy}$ val
    template<typename T1, typename T2, typename T3>
    void buffer_${name}$_const(T1 *A, T2 *B, T3 val, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = B[i] ${sy}$ val;
      }
    }

    template<typename T1, typename T2, typename T3>
    void const_${name}$_buffer(T1 *A, T2 val, T3 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = val ${sy}$ B[i];
      }
    }

    // A = U ${sy}$ V
    template<typename T1, typename T2, typename T3>
    void buffer_${name}$_buffer(T1 *A, T2 *U, T3 *V, int size) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      ///:set a = sy
      ///:if a == '*'
      ///:set a = '%'
      ///:endif
       // CA = CU ${a}$ CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      
#pragma omp parallel for
      for (int i = 0; i < size; i++) {
        A[i] = U[i] ${sy}$ V[i];
      }
      //toc("kernel");
    }

    ///:endfor 

    // A = B
    template<typename T>
    void copy_buffer(T *A, T *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = B[i];
      }
    }

    ///:mute
    ///:include "NodeType.fypp"
    ///:endmute
    ///:for k in [i for i in L if (i[3] == 'C')]
    ///:set name = k[1]
    ///:set sy = k[2]
    ///:set ef = k[7]
    // ans = ${ef}$
    template<typename T1, typename T2>
    void buffer_${name}$(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        ///:if name != 'rcp'
        A[i] = ${sy}$(B[i]);
        ///:else
        A[i] = 1.0 / B[i];
        ///:endif
      }
    }

    ///:endfor

    template<typename T>
    void buffer_pow(double *A, T *B, double m, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = pow(B[i], m);
      }
    }

    template<typename T>
    void buffer_not(int *A, T *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = !(B[i]);
      }
    }

    ///:for k in [['max', '>'], ['min', '<']]
    ///:set name = k[0]
    ///:set sy = k[1]
    template<typename T>
    int3 buffer_${name}$_const(T &val, T *A, Box box, int sw, int size) {
      int x = 0, y = 0, z = 0;
      int xs, xe, ys, ye, zs, ze;
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

      int M = xe - xs;
      int N = ye - ys;
      int K = ze - zs;
      val = A[sw * M * N + sw + M + sw];

      int3 pos = {sw, sw, sw};

      int cnt = 0;
      for (int k = zs; k < ze; k++) {
        for (int j = ys; j < ye; j++) {
          for (int i = xs; i < xe; i++) {
            if ((xs + sw <= i && i < xe - sw) &&
                (ys + sw <= j && j < ye - sw) &&
                (zs + sw <= k && k < ze - sw)) {
              if (A[cnt] ${sy}$ val) {
                val = A[cnt];
                pos = {i, j, k};  
              }
            }
            cnt++;
          }
        }
      }
      return pos;
    }

    ///:endfor

    template<typename T>
    void buffer_sum_scalar_const(T *val, T *A, Box box, int sw, int size) {
      int x = 0, y = 0, z = 0;
      int xs, xe, ys, ye, zs, ze;
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

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
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

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
              if((i == xe - sw -1) && (type == 1 || type == 0))
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
          int find = 0;
          for (int i = xe-1; i >= xs; i--) {
            if ((xs + sw <= i && i < xe - sw) &&
                (ys + sw <= j && j < ye - sw) &&
                (zs + sw <= k && k < ze - sw)) {
              int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
              int temp2 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs-1);
              ap[temp1]+=A[temp1];
              if(i > xs + sw){
                ap[temp2] += ap[temp1];
              }
              if((i == xs + sw) && (type == 1 || type == 2))
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
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

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
              if((j == ye - sw -1) && (type == 1 || type == 0))
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
          int find = 0;
          for (int j = ye-1; j >= ys; j--) {
            if ((xs + sw <= i && i < xe - sw) &&
                (ys + sw <= j && j < ye - sw) &&
                (zs + sw <= k && k < ze - sw)) {
              int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
              int temp2 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-1-ys)+(i-xs);
              ap[temp1]+=A[temp1];
              if(j > ys + sw){
                ap[temp2] += ap[temp1];
              }
              if((j == ys + sw) && (type == 1 || type == 2))
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
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

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
              if((k == ze - sw -1) && (type == 1 || type == 0))
                ap[temp1] = buffer[index++];
              else 
                ap[temp1] = 0;
            }
          }
        }
      }
      index = 0;
      for (int j = ys; j < ye; j++) {
        for (int i = xs; i < xe; i++) {
          int find = 0;
          for (int k = ze-1; k >= zs; k--) {
            if ((xs + sw <= i && i < xe - sw) &&
                (ys + sw <= j && j < ye - sw) &&
                (zs + sw <= k && k < ze - sw)) {
              int temp1 = (xe-xs)*(ye-ys)*(k-zs)+(xe-xs)*(j-ys)+(i-xs);
              int temp2 = (xe-xs)*(ye-ys)*(k-1-zs)+(xe-xs)*(j-ys)+(i-xs);
              ap[temp1]+=A[temp1];
              if(k > zs + sw){
                ap[temp2] += ap[temp1];
              }
              if((k == zs + sw) && (type == 1 || type == 2))
                buffer[index++] = ap[temp1];
            }
          }
        }
      }
    }


  }
}

#endif
