

namespace oa{
  namespace internal{

    ///:for k in [['max','>',''],['min','<',''],['abs_max','>'],['abs_min','<']]
    ///:set name = k[0]
    ///:set sy = k[1]
    template<typename T>
    void buffer_${name}$_const(T &val, int* pos, T *A,
            Box box, int sw) {
      int x = 0, y = 0, z = 0;
      int xs, xe, ys, ye, zs, ze;
      box.get_corners(xs, xe, ys, ye, zs, ze, sw);

      int M = xe - xs;
      int N = ye - ys;
      int K = ze - zs;

      //std::cout<<"MNK:"<<M<<" "<<N<<" "<<" "<<K<<" sw="<<sw<<std::endl;
      
      ///:mute
      ///:if k[0:3] == 'abs'
      ///:set op = "abs"
      ///:else
      ///:set op = ""
      ///:endif
      ///:endmute
      
      val = ${op}$(A[sw * M * N + sw * M + sw]);

      pos[0] = xs;
      pos[1] = ys;
      pos[2] = zs;


      for (int k = sw; k < K-sw; k++) {
        for (int j = sw; j < N-sw; j++) {
          for (int i = sw; i < M-sw; i++) {
            //printf("(%d,%d,%d) = %d"
            // std::cout<<"("<<i<<","<<j<<","<<k<<")="
            //          <<A[i + j * M + k * M * N]<<std::endl;
            
            if (A[i + j * M + k * M * N] ${sy}$ ${op}$(val)) {
              val = A[i + j * M + k * M * N];
              pos[0] = i - sw + xs;
              pos[1] = j - sw + ys;
              pos[2] = k - sw + zs;
            }
          }
        }
      }

    }
    ///:endfor


    ///:for k in [['max2','>'],['min2','<']]
    ///:set name = k[0]
    ///:set sy = k[1]
    template<typename TC, typename TA, typename TB>
    void buffer_${name}$(
        TC* C, TA *A, TB *B,
        const Shape& sc,
        const Shape& sa,
        const Shape& sb,
        const Box& wc,
        const Box& wa,
        const Box& wb) {
      
      int x = 0, y = 0, z = 0;
      int xs_a, xe_a, ys_a, ye_a, zs_a, ze_a;
      wa.get_corners(xs_a, xe_a, ys_a, ye_a, zs_a, ze_a);

      int xs_b, xe_b, ys_b, ye_b, zs_b, ze_b;
      wb.get_corners(xs_b, xe_b, ys_b, ye_b, zs_b, ze_b);

      int xs_c, xe_c, ys_c, ye_c, zs_c, ze_c;
      wc.get_corners(xs_c, xe_c, ys_c, ye_c, zs_c, ze_c);
      
      int M = xe_a - xs_a;
      int N = ye_a - ys_a;
      int P = ze_a - zs_a;

      const int MC = sc[0];
      const int NC = sc[1];
      const int PC = sc[2];

      const int MB = sb[0];
      const int NB = sb[1];
      const int PB = sb[2];

      const int MA = sa[0];
      const int NA = sa[1];
      const int PA = sa[2];

      for(int k = 0; k < P; ++k){
        for(int j = 0; j < N; ++j){
          for(int i = 0; i < M; ++i){
            const int ia = (k + zs_a) * MA * NA +
              (j + ys_a) * MA + (i + xs_a);
            const int ib = (k + zs_b) * MB * NB +
              (j + ys_b) * MB + (i + xs_b);
            const int ic = (k + zs_c) * MC * NC +
              (j + ys_c) * MC + (i + xs_c);
            C[ic] = (A[ia] ${sy}$ B[ib]) ? A[ia] : B[ib];
          }
        }
      }
      //std::cout<<"MNK:"<<M<<" "<<N<<" "<<" "
      //<<K<<" sw="<<sw<<std::endl;
    }
    ///:endfor
  }
}
