
#ifndef __BASIC_INTERNAL_HPP__
#define __BASIC_INTERNAL_HPP__


namespace oa{
  namespace internal{
#define calc_id(i,j,k,S) ((k)*(S[0])*(S[1])+(j)*(S[0])+(i))

    // A = B + val
    template<typename T1, typename T2, typename T3>
    void buffer_plus_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] + val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] + val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_plus_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val + B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val + B[i];
      // }
    }

    // A = U + V
    template<typename T1, typename T2, typename T3>
    void buffer_plus_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU + CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] + V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] + (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo + V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_plus_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] + V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B - val
    template<typename T1, typename T2, typename T3>
    void buffer_minus_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] - val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] - val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_minus_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val - B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val - B[i];
      // }
    }

    // A = U - V
    template<typename T1, typename T2, typename T3>
    void buffer_minus_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU - CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] - V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] - (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo - V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_minus_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] - V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B * val
    template<typename T1, typename T2, typename T3>
    void buffer_mult_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] * val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] * val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_mult_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val * B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val * B[i];
      // }
    }

    // A = U * V
    template<typename T1, typename T2, typename T3>
    void buffer_mult_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU % CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] * V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] * (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo * V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_mult_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] * V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B / val
    template<typename T1, typename T2, typename T3>
    void buffer_divd_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] / val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] / val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_divd_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val / B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val / B[i];
      // }
    }

    // A = U / V
    template<typename T1, typename T2, typename T3>
    void buffer_divd_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU / CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] / V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] / (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo / V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_divd_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] / V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B > val
    template<typename T1, typename T2, typename T3>
    void buffer_gt_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] > val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] > val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_gt_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val > B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val > B[i];
      // }
    }

    // A = U > V
    template<typename T1, typename T2, typename T3>
    void buffer_gt_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU > CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] > V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] > (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo > V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_gt_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] > V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B >= val
    template<typename T1, typename T2, typename T3>
    void buffer_ge_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] >= val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] >= val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_ge_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val >= B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val >= B[i];
      // }
    }

    // A = U >= V
    template<typename T1, typename T2, typename T3>
    void buffer_ge_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU >= CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] >= V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] >= (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo >= V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_ge_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] >= V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B < val
    template<typename T1, typename T2, typename T3>
    void buffer_lt_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] < val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] < val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_lt_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val < B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val < B[i];
      // }
    }

    // A = U < V
    template<typename T1, typename T2, typename T3>
    void buffer_lt_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU < CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] < V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] < (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo < V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_lt_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] < V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B <= val
    template<typename T1, typename T2, typename T3>
    void buffer_le_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] <= val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] <= val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_le_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val <= B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val <= B[i];
      // }
    }

    // A = U <= V
    template<typename T1, typename T2, typename T3>
    void buffer_le_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU <= CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] <= V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] <= (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo <= V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_le_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] <= V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B == val
    template<typename T1, typename T2, typename T3>
    void buffer_eq_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] == val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] == val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_eq_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val == B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val == B[i];
      // }
    }

    // A = U == V
    template<typename T1, typename T2, typename T3>
    void buffer_eq_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU == CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] == V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] == (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo == V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_eq_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] == V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B != val
    template<typename T1, typename T2, typename T3>
    void buffer_ne_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] != val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] != val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_ne_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val != B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val != B[i];
      // }
    }

    // A = U != V
    template<typename T1, typename T2, typename T3>
    void buffer_ne_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU != CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] != V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] != (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo != V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_ne_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] != V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B || val
    template<typename T1, typename T2, typename T3>
    void buffer_or_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] || val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] || val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_or_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val || B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val || B[i];
      // }
    }

    // A = U || V
    template<typename T1, typename T2, typename T3>
    void buffer_or_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU || CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] || V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] || (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo || V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_or_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] || V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    
    // A = B && val
    template<typename T1, typename T2, typename T3>
    void buffer_and_const(T1 *A, T2 *B, T3 val, const Shape& s,
                               int sw)
    {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = B[index] && val;
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = B[i] && val;
      // }
    }



    template<typename T1, typename T2, typename T3>
    void const_and_buffer(T1 *A, T2 val, T3 *B, const Shape& s,
                               int sw) {

      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = val && B[index];
          }

      // for (int i = 0; i < size; i++) {
      //   A[i] = val && B[i];
      // }
    }

    // A = U && V
    template<typename T1, typename T2, typename T3>
    void buffer_and_buffer(T1 *A, T2 *U, T3 *V,
                                const Shape& s,
                                int sw) {

      // arma::Col<T1> CA = oa::utils::make_vec<T1>(size, A);
      // arma::Col<T2> CU = oa::utils::make_vec<T2>(size, U);
      // arma::Col<T3> CV = oa::utils::make_vec<T3>(size, V);

      // CA = CU && CV;
      
      //std::cout<<CA(arma::span(1, 10))<<std::endl;
      
      //tic("kernel");
      const int M = s[0];
      const int N = s[1];
      const int P = s[2];

      for (int k = sw; k < P-sw; k++) 
        for (int j = sw; j < N-sw; j++)      
          for (int i = sw; i < M-sw; i++) {
            const int index = i + j * M + k * M * N;
            A[index] = U[index] && V[index];
          }

      // #pragma omp parallel for
      //       for (int i = 0; i < size; i++) {
      //         A[i] = (T1)U[i] && (T1)V[i];
      //       }
      //toc("kernel");
    }




    // A = U pesudo && V
    template<typename T1, typename T2, typename T3>
    void pseudo_buffer_and_buffer(T1 *A, T2 *U, T3 *V, 
                                       Box ab, Box ub, Box vb, Shape Sa, Shape Su, Shape Sv, int sw) {

      oa_int3 a_x = calc_step(ab, 0, sw);
      oa_int3 a_y = calc_step(ab, 1, sw);
      oa_int3 a_z = calc_step(ab, 2, sw);

      oa_int3 u_x = calc_step(ub, 0, sw);
      oa_int3 u_y = calc_step(ub, 1, sw);
      oa_int3 u_z = calc_step(ub, 2, sw);

      oa_int3 v_x = calc_step(vb, 0, sw);
      oa_int3 v_y = calc_step(vb, 1, sw);
      oa_int3 v_z = calc_step(vb, 2, sw);

      
      
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
              U[calc_id(iu, ju, ku, Su)] && V[calc_id(iv, jv, kv, Sv)];
          }
        }
      }
    }
    

    // ans = exp(A)
    template<typename T1, typename T2>
    void buffer_exp(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = exp(B[i]);
      }
    }

    // ans = sin(A)
    template<typename T1, typename T2>
    void buffer_sin(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = sin(B[i]);
      }
    }

    // ans = tan(A)
    template<typename T1, typename T2>
    void buffer_tan(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = tan(B[i]);
      }
    }

    // ans = cos(A)
    template<typename T1, typename T2>
    void buffer_cos(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = cos(B[i]);
      }
    }

    // ans = 1.0/A
    template<typename T1, typename T2>
    void buffer_rcp(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = 1.0 / B[i];
      }
    }

    // ans = sqrt(A)
    template<typename T1, typename T2>
    void buffer_sqrt(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = sqrt(B[i]);
      }
    }

    // ans = asin(A)
    template<typename T1, typename T2>
    void buffer_asin(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = asin(B[i]);
      }
    }

    // ans = acos(A)
    template<typename T1, typename T2>
    void buffer_acos(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = acos(B[i]);
      }
    }

    // ans = atan(A)
    template<typename T1, typename T2>
    void buffer_atan(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = atan(B[i]);
      }
    }

    // ans = abs(A)
    template<typename T1, typename T2>
    void buffer_abs(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = abs(B[i]);
      }
    }

    // ans = log(A)
    template<typename T1, typename T2>
    void buffer_log(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = log(B[i]);
      }
    }

    // ans = +(A)
    template<typename T1, typename T2>
    void buffer_uplus(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = +(B[i]);
      }
    }

    // ans = -(A)
    template<typename T1, typename T2>
    void buffer_uminus(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = -(B[i]);
      }
    }

    // ans = log10(A)
    template<typename T1, typename T2>
    void buffer_log10(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = log10(B[i]);
      }
    }

    // ans = tanh(A)
    template<typename T1, typename T2>
    void buffer_tanh(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = tanh(B[i]);
      }
    }

    // ans = sinh(A)
    template<typename T1, typename T2>
    void buffer_sinh(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = sinh(B[i]);
      }
    }

    // ans = cosh(A)
    template<typename T1, typename T2>
    void buffer_cosh(T1 *A, T2 *B, int size) {
      for (int i = 0; i < size; i++) {
        A[i] = cosh(B[i]);
      }
    }



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
