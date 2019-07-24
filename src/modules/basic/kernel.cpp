
#include "kernel.hpp"
#include "../../Array.hpp"
#include "kernel.hpp"
#include "internal.hpp"


namespace oa{
  namespace kernel{
    
    // crate kernel_plus
    // A = U + V
    ArrayPtr kernel_plus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_plus<int, int, int>;
        kernel_table[10] = t_kernel_plus<float, int, float>;
        kernel_table[12] = t_kernel_plus<float, float, int>;
        kernel_table[13] = t_kernel_plus<float, float, float>;
        kernel_table[20] = t_kernel_plus<double, int, double>;
        kernel_table[23] = t_kernel_plus<double, float, double>;
        kernel_table[24] = t_kernel_plus<double, double, int>;
        kernel_table[25] = t_kernel_plus<double, double, float>;
        kernel_table[26] = t_kernel_plus<double, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_minus
    // A = U - V
    ArrayPtr kernel_minus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_minus<int, int, int>;
        kernel_table[10] = t_kernel_minus<float, int, float>;
        kernel_table[12] = t_kernel_minus<float, float, int>;
        kernel_table[13] = t_kernel_minus<float, float, float>;
        kernel_table[20] = t_kernel_minus<double, int, double>;
        kernel_table[23] = t_kernel_minus<double, float, double>;
        kernel_table[24] = t_kernel_minus<double, double, int>;
        kernel_table[25] = t_kernel_minus<double, double, float>;
        kernel_table[26] = t_kernel_minus<double, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_mult
    // A = U * V
    ArrayPtr kernel_mult(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_mult<int, int, int>;
        kernel_table[10] = t_kernel_mult<float, int, float>;
        kernel_table[12] = t_kernel_mult<float, float, int>;
        kernel_table[13] = t_kernel_mult<float, float, float>;
        kernel_table[20] = t_kernel_mult<double, int, double>;
        kernel_table[23] = t_kernel_mult<double, float, double>;
        kernel_table[24] = t_kernel_mult<double, double, int>;
        kernel_table[25] = t_kernel_mult<double, double, float>;
        kernel_table[26] = t_kernel_mult<double, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_divd
    // A = U / V
    ArrayPtr kernel_divd(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_divd<int, int, int>;
        kernel_table[10] = t_kernel_divd<float, int, float>;
        kernel_table[12] = t_kernel_divd<float, float, int>;
        kernel_table[13] = t_kernel_divd<float, float, float>;
        kernel_table[20] = t_kernel_divd<double, int, double>;
        kernel_table[23] = t_kernel_divd<double, float, double>;
        kernel_table[24] = t_kernel_divd<double, double, int>;
        kernel_table[25] = t_kernel_divd<double, double, float>;
        kernel_table[26] = t_kernel_divd<double, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    

    // crate kernel_gt
    // A = U > V
    ArrayPtr kernel_gt(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_gt<int, int, int>;
        kernel_table[10] = t_kernel_gt<int, int, float>;
        kernel_table[12] = t_kernel_gt<int, float, int>;
        kernel_table[13] = t_kernel_gt<int, float, float>;
        kernel_table[20] = t_kernel_gt<int, int, double>;
        kernel_table[23] = t_kernel_gt<int, float, double>;
        kernel_table[24] = t_kernel_gt<int, double, int>;
        kernel_table[25] = t_kernel_gt<int, double, float>;
        kernel_table[26] = t_kernel_gt<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_ge
    // A = U >= V
    ArrayPtr kernel_ge(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_ge<int, int, int>;
        kernel_table[10] = t_kernel_ge<int, int, float>;
        kernel_table[12] = t_kernel_ge<int, float, int>;
        kernel_table[13] = t_kernel_ge<int, float, float>;
        kernel_table[20] = t_kernel_ge<int, int, double>;
        kernel_table[23] = t_kernel_ge<int, float, double>;
        kernel_table[24] = t_kernel_ge<int, double, int>;
        kernel_table[25] = t_kernel_ge<int, double, float>;
        kernel_table[26] = t_kernel_ge<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_lt
    // A = U < V
    ArrayPtr kernel_lt(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_lt<int, int, int>;
        kernel_table[10] = t_kernel_lt<int, int, float>;
        kernel_table[12] = t_kernel_lt<int, float, int>;
        kernel_table[13] = t_kernel_lt<int, float, float>;
        kernel_table[20] = t_kernel_lt<int, int, double>;
        kernel_table[23] = t_kernel_lt<int, float, double>;
        kernel_table[24] = t_kernel_lt<int, double, int>;
        kernel_table[25] = t_kernel_lt<int, double, float>;
        kernel_table[26] = t_kernel_lt<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_le
    // A = U <= V
    ArrayPtr kernel_le(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_le<int, int, int>;
        kernel_table[10] = t_kernel_le<int, int, float>;
        kernel_table[12] = t_kernel_le<int, float, int>;
        kernel_table[13] = t_kernel_le<int, float, float>;
        kernel_table[20] = t_kernel_le<int, int, double>;
        kernel_table[23] = t_kernel_le<int, float, double>;
        kernel_table[24] = t_kernel_le<int, double, int>;
        kernel_table[25] = t_kernel_le<int, double, float>;
        kernel_table[26] = t_kernel_le<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_eq
    // A = U == V
    ArrayPtr kernel_eq(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_eq<int, int, int>;
        kernel_table[10] = t_kernel_eq<int, int, float>;
        kernel_table[12] = t_kernel_eq<int, float, int>;
        kernel_table[13] = t_kernel_eq<int, float, float>;
        kernel_table[20] = t_kernel_eq<int, int, double>;
        kernel_table[23] = t_kernel_eq<int, float, double>;
        kernel_table[24] = t_kernel_eq<int, double, int>;
        kernel_table[25] = t_kernel_eq<int, double, float>;
        kernel_table[26] = t_kernel_eq<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_ne
    // A = U != V
    ArrayPtr kernel_ne(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_ne<int, int, int>;
        kernel_table[10] = t_kernel_ne<int, int, float>;
        kernel_table[12] = t_kernel_ne<int, float, int>;
        kernel_table[13] = t_kernel_ne<int, float, float>;
        kernel_table[20] = t_kernel_ne<int, int, double>;
        kernel_table[23] = t_kernel_ne<int, float, double>;
        kernel_table[24] = t_kernel_ne<int, double, int>;
        kernel_table[25] = t_kernel_ne<int, double, float>;
        kernel_table[26] = t_kernel_ne<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_or
    // A = U || V
    ArrayPtr kernel_or(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_or<int, int, int>;
        kernel_table[10] = t_kernel_or<int, int, float>;
        kernel_table[12] = t_kernel_or<int, float, int>;
        kernel_table[13] = t_kernel_or<int, float, float>;
        kernel_table[20] = t_kernel_or<int, int, double>;
        kernel_table[23] = t_kernel_or<int, float, double>;
        kernel_table[24] = t_kernel_or<int, double, int>;
        kernel_table[25] = t_kernel_or<int, double, float>;
        kernel_table[26] = t_kernel_or<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }

    // crate kernel_and
    // A = U && V
    ArrayPtr kernel_and(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        //create kernel_table
        kernel_table[0] = t_kernel_and<int, int, int>;
        kernel_table[10] = t_kernel_and<int, int, float>;
        kernel_table[12] = t_kernel_and<int, float, int>;
        kernel_table[13] = t_kernel_and<int, float, float>;
        kernel_table[20] = t_kernel_and<int, int, double>;
        kernel_table[23] = t_kernel_and<int, float, double>;
        kernel_table[24] = t_kernel_and<int, double, int>;
        kernel_table[25] = t_kernel_and<int, double, float>;
        kernel_table[26] = t_kernel_and<int, double, double>;
        has_init = true;
      }

      int case_num = dt * 9 + u_dt * 3 + v_dt;
      ap = kernel_table[case_num](ops_ap);
      return ap;
    }



    ///!:for k in [i for i in L if (i[3] == 'C')]
    // crate kernel_exp
    // A = exp(A)
    ArrayPtr kernel_exp(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_exp(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_exp(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_exp(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_sin
    // A = sin(A)
    ArrayPtr kernel_sin(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_sin(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_sin(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_sin(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_tan
    // A = tan(A)
    ArrayPtr kernel_tan(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_tan(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_tan(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_tan(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_cos
    // A = cos(A)
    ArrayPtr kernel_cos(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_cos(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_cos(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_cos(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_rcp
    // A = 1.0/A
    ArrayPtr kernel_rcp(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_rcp(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_rcp(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_rcp(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_sqrt
    // A = sqrt(A)
    ArrayPtr kernel_sqrt(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_sqrt(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_sqrt(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_sqrt(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_asin
    // A = asin(A)
    ArrayPtr kernel_asin(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_asin(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_asin(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_asin(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_acos
    // A = acos(A)
    ArrayPtr kernel_acos(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_acos(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_acos(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_acos(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_atan
    // A = atan(A)
    ArrayPtr kernel_atan(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_atan(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_atan(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_atan(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_abs
    // A = abs(A)
    ArrayPtr kernel_abs(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_abs(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_abs(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_abs(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_log
    // A = log(A)
    ArrayPtr kernel_log(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_log(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_log(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_log(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_uplus
    // A = +(A)
    ArrayPtr kernel_uplus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_uplus(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_uplus(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_uplus(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_uminus
    // A = -(A)
    ArrayPtr kernel_uminus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_uminus(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_uminus(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_uminus(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_log10
    // A = log10(A)
    ArrayPtr kernel_log10(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_log10(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_log10(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_log10(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_tanh
    // A = tanh(A)
    ArrayPtr kernel_tanh(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_tanh(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_tanh(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_tanh(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_sinh
    // A = sinh(A)
    ArrayPtr kernel_sinh(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_sinh(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_sinh(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_sinh(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }

    // crate kernel_cosh
    // A = cosh(A)
    ArrayPtr kernel_cosh(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      
      int u_dt = u->get_data_type();

      switch(u_dt) {
      case DATA_INT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);    
        oa::internal::buffer_cosh(
            (int *) ap->get_buffer(),
            (int *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_FLOAT:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_FLOAT);        
        oa::internal::buffer_cosh(
            (float *) ap->get_buffer(),
            (float *) u->get_buffer(),
            ap->buffer_size());
        break;
      case DATA_DOUBLE:
        ap = ArrayPool::global()->get(u->get_partition(), DATA_DOUBLE);    
        oa::internal::buffer_cosh(
            (double *) ap->get_buffer(),
            (double *) u->get_buffer(),
            ap->buffer_size());
        break;
      }
      return ap;
    }


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
  }
}
