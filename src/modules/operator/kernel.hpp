
#ifndef __OP_KERNEL_HPP__
#define __OP_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "internal.hpp"
#include "../../Grid.hpp"
#include <vector>
using namespace std;


namespace oa {
  namespace kernel {

    // return ANS = DXC(A)
    ArrayPtr kernel_dxc(vector<ArrayPtr> &ops_ap);
    // return ANS = DYC(A)
    ArrayPtr kernel_dyc(vector<ArrayPtr> &ops_ap);
    // return ANS = DZC(A)
    ArrayPtr kernel_dzc(vector<ArrayPtr> &ops_ap);
    // return ANS = AXB(A)
    ArrayPtr kernel_axb(vector<ArrayPtr> &ops_ap);
    // return ANS = AXF(A)
    ArrayPtr kernel_axf(vector<ArrayPtr> &ops_ap);
    // return ANS = AYB(A)
    ArrayPtr kernel_ayb(vector<ArrayPtr> &ops_ap);
    // return ANS = AYF(A)
    ArrayPtr kernel_ayf(vector<ArrayPtr> &ops_ap);
    // return ANS = AZB(A)
    ArrayPtr kernel_azb(vector<ArrayPtr> &ops_ap);
    // return ANS = AZF(A)
    ArrayPtr kernel_azf(vector<ArrayPtr> &ops_ap);
    // return ANS = DXB(A)
    ArrayPtr kernel_dxb(vector<ArrayPtr> &ops_ap);
    // return ANS = DXF(A)
    ArrayPtr kernel_dxf(vector<ArrayPtr> &ops_ap);
    // return ANS = DYB(A)
    ArrayPtr kernel_dyb(vector<ArrayPtr> &ops_ap);
    // return ANS = DYF(A)
    ArrayPtr kernel_dyf(vector<ArrayPtr> &ops_ap);
    // return ANS = DZB(A)
    ArrayPtr kernel_dzb(vector<ArrayPtr> &ops_ap);
    // return ANS = DZF(A)
    ArrayPtr kernel_dzf(vector<ArrayPtr> &ops_ap);

    // crate kernel_dxc
    // A = dxc(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxc_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXC);

      // printf("pos:%d, TYPE_DXC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxc_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxc_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dyc
    // A = dyc(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyc_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYC);

      // printf("pos:%d, TYPE_DYC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyc_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyc_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dzc
    // A = dzc(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzc_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzc...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZC);

      // printf("pos:%d, TYPE_DZC, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzc_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzc_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_axb
    // A = axb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXB);

      // printf("pos:%d, TYPE_AXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_axf
    // A = axf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_axf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel axf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AXF);

      // printf("pos:%d, TYPE_AXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::axf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::axf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_ayb
    // A = ayb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYB);

      // printf("pos:%d, TYPE_AYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_ayf
    // A = ayf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_ayf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel ayf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AYF);

      // printf("pos:%d, TYPE_AYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::ayf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::ayf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_azb
    // A = azb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZB);

      // printf("pos:%d, TYPE_AZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_azf
    // A = azf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_azf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel azf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_AZF);

      // printf("pos:%d, TYPE_AZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::azf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::azf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dxb
    // A = dxb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXB);

      // printf("pos:%d, TYPE_DXB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dxf
    // A = dxf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dxf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dxf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{1, 0, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DXF);

      // printf("pos:%d, TYPE_DXF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 0, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 0);
#endif
          oa::internal::dxf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dxf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dyb
    // A = dyb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYB);

      // printf("pos:%d, TYPE_DYB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dyf
    // A = dyf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dyf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dyf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 1, 0}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DYF);

      // printf("pos:%d, TYPE_DYF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 1, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 1);
#endif
          oa::internal::dyf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dyf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dzb
    // A = dzb(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzb_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzb...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      lbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZB);

      // printf("pos:%d, TYPE_DZB, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzb_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzb_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // crate kernel_dzf
    // A = dzf(U)

    
    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_ooo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_ooz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_ooz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_ooz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_ooz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_ooz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_oyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_oyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_oyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_oyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_oyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_oyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_xoo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_xoz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xoz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xoz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xoz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xoz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_xyo(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyo"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyo"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyo_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyo_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyo_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyo_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyo_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyo_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    template<typename T1, typename T2>
    ArrayPtr t_kernel_dzf_with_grid_xyz(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr ap;

      // // use pseudo
      // if (u->is_pseudo()) {
      //   if (u->has_pseudo_3d() == false)
      //     u->set_pseudo_3d(oa::funcs::make_psudo3d(u));
      //   u = u->get_pseudo_3d();
      // }

      // printf("calling kernel dzf...\n");
      int dt = oa::utils::to_type<T1>();

      ap = ArrayPool::global()->get(u->get_partition(), dt);
      ap->set_bitset(u->get_bitset());

      int sw = u->get_partition()->get_stencil_width();
      Shape sp = u->local_shape();
      Shape S = u->buffer_shape();

      oa_int3 lbound = {{0, 0, 0}};
      oa_int3 rbound = {{0, 0, 0}};
      rbound = {{0, 0, 1}};

      
#ifndef __HAVE_CUDA__
      T1* ans = (T1*) ap->get_buffer();
      T2* buffer = (T2*) u->get_buffer();
#else
      ap->memcopy_gpu_to_cpu();
      T1* ans = (T1*) ap->get_cpu_buffer();
#endif

      /*
        to chose the right bind grid
      */

      // get_gridptr
      ArrayPtr gridptr = Grid::global()->get_grid(u->get_pos(), TYPE_DZF);

      // printf("pos:%d, TYPE_DZF, %d\n", u->get_pos(), gridptr == NULL);
      // default grid data type
      int grid_dt = DATA_FLOAT;
      void* grid_buffer = NULL;
      Shape SG = {0, 0, 0};

      // cout<<"with_grid_xyz"<<endl; 
      if (gridptr != NULL) {
        // gridptr->display("gridptr = ");
        // cout<<"not null"<<endl;
        // cout<<"with_grid_xyz"<<endl;
        grid_dt = gridptr->get_data_type();
#ifndef __HAVE_CUDA__
        grid_buffer = gridptr->get_buffer();
#else
        gridptr->memcopy_gpu_to_cpu();
        grid_buffer = gridptr->get_cpu_buffer();
#endif
        SG = gridptr->buffer_shape();
      }

      vector<MPI_Request> reqs;
      int cnt = 0;
      switch(grid_dt) {
      case DATA_INT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyz_calc_inside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyz_calc_outside<T1, T2, int>(ans,
                  buffer, (int*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_FLOAT:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyz_calc_inside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyz_calc_outside<T1, T2, float>(ans,
                  buffer, (float*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      case DATA_DOUBLE:
      {
#ifdef __HAVE_CUDA__
          u->memcopy_gpu_to_cpu();
          T2* buffer = (T2*) u->get_cpu_buffer();
          oa::funcs::update_ghost_start(u, reqs, 2, {{0,0,0}}, {{0,0,0}}, CPU);
#else
          oa::funcs::update_ghost_start(u, reqs, 2);
#endif
          oa::internal::dzf_with_grid_xyz_calc_inside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          oa::funcs::update_ghost_end(reqs);
          oa::internal::dzf_with_grid_xyz_calc_outside<T1, T2, double>(ans,
                  buffer, (double*)grid_buffer, lbound, rbound, sw, sp, S, SG);
          break;
      }
      }

#ifdef __HAVE_CUDA__
      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }


  }
}
#endif
