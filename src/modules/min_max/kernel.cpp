
#include "kernel.hpp"
  
  






namespace oa{
  namespace kernel{

    // crate kernel_min
    // A = min(A)
    ArrayPtr kernel_min(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_min<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_min<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_min<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_max
    // A = max(A)
    ArrayPtr kernel_max(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_max<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_max<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_max<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_min_at
    // A = min_at(A)
    ArrayPtr kernel_min_at(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_min_at<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_min_at<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_min_at<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_max_at
    // A = max_at(A)
    ArrayPtr kernel_max_at(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_max_at<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_max_at<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_max_at<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_abs_max
    // A = abs_max(A)
    ArrayPtr kernel_abs_max(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_abs_max<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_abs_max<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_abs_max<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_abs_min
    // A = abs_min(A)
    ArrayPtr kernel_abs_min(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_abs_min<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_abs_min<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_abs_min<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_abs_max_at
    // A = abs_max_at(A)
    ArrayPtr kernel_abs_max_at(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_abs_max_at<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_abs_max_at<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_abs_max_at<double>(ops_ap);
        break;
      }
      return ap;
    }
    // crate kernel_abs_min_at
    // A = abs_min_at(A)
    ArrayPtr kernel_abs_min_at(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;
      int u_dt = u->get_data_type();
      switch(u_dt) {
      case DATA_INT:
        ap = t_kernel_abs_min_at<int>(ops_ap);
        break;
      case DATA_FLOAT:
        ap = t_kernel_abs_min_at<float>(ops_ap);
        break;
      case DATA_DOUBLE:
        ap = t_kernel_abs_min_at<double>(ops_ap);
        break;
      }
      return ap;
    }


    ArrayPtr kernel_min2(vector<ArrayPtr> &ops_ap) {

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        kernel_table[0] =
          t_kernel_min2<int, int, int>;
        kernel_table[10] =
          t_kernel_min2<float, int, float>;
        kernel_table[12] =
          t_kernel_min2<float, float, int>;
        kernel_table[13] =
          t_kernel_min2<float, float, float>;
        kernel_table[20] =
          t_kernel_min2<double, int, double>;
        kernel_table[23] =
          t_kernel_min2<double, float, double>;
        kernel_table[24] =
          t_kernel_min2<double, double, int>;
        kernel_table[25] =
          t_kernel_min2<double, double, float>;
        kernel_table[26] =
          t_kernel_min2<double, double, double>;
        has_init = true;
      }

      const ArrayPtr& u = ops_ap[0];
      const ArrayPtr& v = ops_ap[1];
      
      const int u_dt = u->get_data_type();
      const int v_dt = u->get_data_type();
      const int r_dt = oa::utils::cast_data_type(u_dt, v_dt);
      int case_num = r_dt * 9 + u_dt * 3 + v_dt;

      
      ArrayPtr ap = kernel_table[case_num](ops_ap);
      return ap;
    }
    ArrayPtr kernel_max2(vector<ArrayPtr> &ops_ap) {

      static bool has_init = false;
      static KernelPtr kernel_table[27];

      if (!has_init) {
        kernel_table[0] =
          t_kernel_max2<int, int, int>;
        kernel_table[10] =
          t_kernel_max2<float, int, float>;
        kernel_table[12] =
          t_kernel_max2<float, float, int>;
        kernel_table[13] =
          t_kernel_max2<float, float, float>;
        kernel_table[20] =
          t_kernel_max2<double, int, double>;
        kernel_table[23] =
          t_kernel_max2<double, float, double>;
        kernel_table[24] =
          t_kernel_max2<double, double, int>;
        kernel_table[25] =
          t_kernel_max2<double, double, float>;
        kernel_table[26] =
          t_kernel_max2<double, double, double>;
        has_init = true;
      }

      const ArrayPtr& u = ops_ap[0];
      const ArrayPtr& v = ops_ap[1];
      
      const int u_dt = u->get_data_type();
      const int v_dt = u->get_data_type();
      const int r_dt = oa::utils::cast_data_type(u_dt, v_dt);
      int case_num = r_dt * 9 + u_dt * 3 + v_dt;

      
      ArrayPtr ap = kernel_table[case_num](ops_ap);
      return ap;
    }

  }
}
