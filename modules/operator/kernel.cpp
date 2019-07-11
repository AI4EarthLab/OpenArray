
#include "kernel.hpp"
#include "../../Array.hpp"
#include "internal.hpp"
#include "../../Grid.hpp"
#include <bitset>


namespace oa {
  namespace kernel {
    // crate kernel_dxc
    ArrayPtr kernel_dxc(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dxc_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dxc_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dxc_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dxc_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dxc_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dxc_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dxc_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dxc_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dxc_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dxc_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dxc_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dxc_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dxc_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dxc_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dxc_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dxc_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dxc_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dxc_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dxc_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dxc_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dxc_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dxc_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dxc_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dxc_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DXC)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DXC on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[0] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dyc
    ArrayPtr kernel_dyc(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dyc_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dyc_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dyc_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dyc_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dyc_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dyc_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dyc_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dyc_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dyc_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dyc_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dyc_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dyc_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dyc_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dyc_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dyc_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dyc_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dyc_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dyc_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dyc_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dyc_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dyc_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dyc_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dyc_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dyc_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DYC)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DYC on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[1] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dzc
    ArrayPtr kernel_dzc(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dzc_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dzc_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dzc_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dzc_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dzc_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dzc_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dzc_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dzc_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dzc_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dzc_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dzc_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dzc_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dzc_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dzc_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dzc_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dzc_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dzc_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dzc_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dzc_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dzc_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dzc_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dzc_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dzc_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dzc_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DZC)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DZC on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[2] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_axb
    ArrayPtr kernel_axb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_axb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_axb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_axb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_axb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_axb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_axb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_axb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_axb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_axb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_axb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_axb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_axb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_axb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_axb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_axb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_axb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_axb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_axb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_axb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_axb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_axb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_axb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_axb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_axb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AXB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AXB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[0] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_axf
    ArrayPtr kernel_axf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_axf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_axf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_axf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_axf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_axf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_axf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_axf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_axf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_axf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_axf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_axf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_axf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_axf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_axf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_axf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_axf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_axf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_axf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_axf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_axf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_axf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_axf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_axf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_axf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AXF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AXF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[0] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_ayb
    ArrayPtr kernel_ayb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_ayb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_ayb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_ayb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_ayb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_ayb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_ayb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_ayb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_ayb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_ayb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_ayb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_ayb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_ayb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_ayb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_ayb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_ayb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_ayb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_ayb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_ayb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_ayb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_ayb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_ayb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_ayb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_ayb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_ayb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AYB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AYB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[1] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_ayf
    ArrayPtr kernel_ayf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_ayf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_ayf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_ayf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_ayf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_ayf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_ayf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_ayf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_ayf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_ayf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_ayf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_ayf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_ayf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_ayf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_ayf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_ayf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_ayf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_ayf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_ayf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_ayf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_ayf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_ayf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_ayf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_ayf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_ayf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AYF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AYF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[1] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_azb
    ArrayPtr kernel_azb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_azb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_azb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_azb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_azb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_azb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_azb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_azb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_azb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_azb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_azb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_azb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_azb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_azb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_azb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_azb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_azb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_azb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_azb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_azb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_azb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_azb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_azb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_azb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_azb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AZB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AZB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[2] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_azf
    ArrayPtr kernel_azf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_azf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_azf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_azf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_azf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_azf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_azf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_azf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_azf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_azf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_azf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_azf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_azf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_azf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_azf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_azf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_azf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_azf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_azf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_azf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_azf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_azf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_azf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_azf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_azf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_AZF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform AZF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[2] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dxb
    ArrayPtr kernel_dxb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dxb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dxb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dxb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dxb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dxb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dxb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dxb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dxb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dxb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dxb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dxb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dxb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dxb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dxb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dxb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dxb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dxb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dxb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dxb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dxb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dxb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dxb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dxb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dxb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DXB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DXB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[0] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dxf
    ArrayPtr kernel_dxf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dxf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dxf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dxf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dxf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dxf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dxf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dxf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dxf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dxf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dxf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dxf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dxf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dxf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dxf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dxf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dxf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dxf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dxf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dxf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dxf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dxf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dxf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dxf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dxf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DXF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DXF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[0] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dyb
    ArrayPtr kernel_dyb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dyb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dyb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dyb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dyb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dyb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dyb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dyb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dyb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dyb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dyb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dyb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dyb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dyb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dyb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dyb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dyb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dyb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dyb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dyb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dyb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dyb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dyb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dyb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dyb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DYB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DYB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[1] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dyf
    ArrayPtr kernel_dyf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dyf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dyf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dyf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dyf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dyf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dyf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dyf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dyf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dyf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dyf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dyf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dyf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dyf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dyf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dyf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dyf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dyf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dyf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dyf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dyf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dyf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dyf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dyf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dyf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DYF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DYF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[1] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dzb
    ArrayPtr kernel_dzb(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dzb_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dzb_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dzb_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dzb_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dzb_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dzb_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dzb_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dzb_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dzb_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dzb_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dzb_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dzb_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dzb_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dzb_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dzb_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dzb_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dzb_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dzb_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dzb_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dzb_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dzb_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dzb_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dzb_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dzb_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DZB)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DZB on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[2] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

    // crate kernel_dzf
    ArrayPtr kernel_dzf(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      
      static bool has_init = false;
      static KernelPtr kernel_table[3][8];

      if (!has_init) {
        //create kernel_table
        
        kernel_table[0][0] =
          t_kernel_dzf_with_grid_ooo<float, int>;
        kernel_table[0][1] =
          t_kernel_dzf_with_grid_ooz<float, int>;
        kernel_table[0][2] =
          t_kernel_dzf_with_grid_oyo<float, int>;
        kernel_table[0][3] =
          t_kernel_dzf_with_grid_oyz<float, int>;
        kernel_table[0][4] =
          t_kernel_dzf_with_grid_xoo<float, int>;
        kernel_table[0][5] =
          t_kernel_dzf_with_grid_xoz<float, int>;
        kernel_table[0][6] =
          t_kernel_dzf_with_grid_xyo<float, int>;
        kernel_table[0][7] =
          t_kernel_dzf_with_grid_xyz<float, int>;

        kernel_table[1][0] =
          t_kernel_dzf_with_grid_ooo<float, float>;
        kernel_table[1][1] =
          t_kernel_dzf_with_grid_ooz<float, float>;
        kernel_table[1][2] =
          t_kernel_dzf_with_grid_oyo<float, float>;
        kernel_table[1][3] =
          t_kernel_dzf_with_grid_oyz<float, float>;
        kernel_table[1][4] =
          t_kernel_dzf_with_grid_xoo<float, float>;
        kernel_table[1][5] =
          t_kernel_dzf_with_grid_xoz<float, float>;
        kernel_table[1][6] =
          t_kernel_dzf_with_grid_xyo<float, float>;
        kernel_table[1][7] =
          t_kernel_dzf_with_grid_xyz<float, float>;

        kernel_table[2][0] =
          t_kernel_dzf_with_grid_ooo<double, double>;
        kernel_table[2][1] =
          t_kernel_dzf_with_grid_ooz<double, double>;
        kernel_table[2][2] =
          t_kernel_dzf_with_grid_oyo<double, double>;
        kernel_table[2][3] =
          t_kernel_dzf_with_grid_oyz<double, double>;
        kernel_table[2][4] =
          t_kernel_dzf_with_grid_xoo<double, double>;
        kernel_table[2][5] =
          t_kernel_dzf_with_grid_xoz<double, double>;
        kernel_table[2][6] =
          t_kernel_dzf_with_grid_xyo<double, double>;
        kernel_table[2][7] =
          t_kernel_dzf_with_grid_xyz<double, double>;

        has_init = true;
      }

      int id = 0;
      int pos = u->get_pos();
      if (pos != -1) {
        bitset<3> bit =
          Grid::global()->get_grid(pos, TYPE_DZF)->get_bitset();
        id = (int)(bit.to_ulong());
      }

      Shape us = u->shape();

      std::string err_msg =
        "unable to perform DZF on array of shape (" +
        to_string(us[0]) + "," + 
        to_string(us[1]) + "," + 
        to_string(us[2]) + ")";
        
      if(us[2] < 2) assert(false && err_msg.c_str());
      

      // printf("id = %d\n", id);
      ap = kernel_table[u_dt][id](ops_ap);

      return ap;
    }

  }
}
