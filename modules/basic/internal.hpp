
#ifndef __BASIC_INTERNAL_HPP__
#define __BASIC_INTERNAL_HPP__

///:mute
///:include "../../NodeType.fypp"
///:endmute

namespace oa{
  namespace internal{
#define calc_id(i,j,k,S) ((k)*(S[0])*(S[1])+(j)*(S[0])+(i))

    ///:for k in [i for i in L if (i[3] == 'A' or i[3] == 'B' or i[3] == 'F')]
    ///:set name = k[1]
    ///:set sy = k[2]
    // A = B ${sy}$ val
    template<typename T1, typename T2, typename T3>
    void buffer_${name}$_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] ${sy}$ val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] ${sy}$ val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_${name}$_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val ${sy}$ B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val ${sy}$ B[i];
      // }
    }

    // A = U ${sy}$ V
    template<typename T1, typename T2, typename T3>
    void buffer_${name}$_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

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
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] ${sy}$ V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] ${sy}$ (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo ${sy}$ V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_${name}$_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      int3 a_x = calc_step(ab, 0, sw);
      int3 a_y = calc_step(ab, 1, sw);
      int3 a_z = calc_step(ab, 2, sw);

      int3 u_x = calc_step(ub, 0, sw);
      int3 u_y = calc_step(ub, 1, sw);
      int3 u_z = calc_step(ub, 2, sw);

      int3 v_x = calc_step(vb, 0, sw);
      int3 v_y = calc_step(vb, 1, sw);
      int3 v_z = calc_step(vb, 2, sw);

      
      
      for (int ka = a_z[0], ku = u_z[0], kv = v_z[0]; 
           ka < a_z[2] && ku < u_z[2] && kv < v_z[2];
           ka += a_z[1], ku += u_z[1], kv += v_z[1]) {
        
        for (int ja = a_y[0], ju = u_y[0], jv = v_y[0];
             ja < a_y[2] && ju < u_y[2] && jv < v_y[2];
             ja += a_y[1], ju += u_y[1], jv += v_y[1]) {

          for (int ia = a_x[0], iu = u_x[0], iv = v_x[0];
               ia < a_x[2] && iu < u_x[2] && iv < v_x[2];
               ia += a_x[1], iu += u_x[1], iv += v_x[1]) {

            // printf("%d %d %d %d %d %d %d\n", ia, ja, ka, Sa[0], Sa[1], Sa[2], 
            //   calc_id(ia, ja, ka, Sa));
            A[calc_id(ia, ja, ka, Sa)] = 
              U[calc_id(iu, ju, ku, Su)] ${sy}$ V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    ///:endfor

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

  }
}

#endif
