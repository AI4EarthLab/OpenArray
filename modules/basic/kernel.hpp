#ifndef __BASIC_KERNEL_HPP__
#define __BASIC_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../op_define.hpp"
#include "internal.hpp"
#include <vector>
using namespace std;
namespace inl = oa::internal;

  
  






namespace oa{
  namespace kernel{

    // crate kernel_plus
    ArrayPtr kernel_plus(vector<ArrayPtr> &ops_ap);
    // crate kernel_minus
    ArrayPtr kernel_minus(vector<ArrayPtr> &ops_ap);
    // crate kernel_mult
    ArrayPtr kernel_mult(vector<ArrayPtr> &ops_ap);
    // crate kernel_divd
    ArrayPtr kernel_divd(vector<ArrayPtr> &ops_ap);
    // crate kernel_gt
    ArrayPtr kernel_gt(vector<ArrayPtr> &ops_ap);
    // crate kernel_ge
    ArrayPtr kernel_ge(vector<ArrayPtr> &ops_ap);
    // crate kernel_lt
    ArrayPtr kernel_lt(vector<ArrayPtr> &ops_ap);
    // crate kernel_le
    ArrayPtr kernel_le(vector<ArrayPtr> &ops_ap);
    // crate kernel_eq
    ArrayPtr kernel_eq(vector<ArrayPtr> &ops_ap);
    // crate kernel_ne
    ArrayPtr kernel_ne(vector<ArrayPtr> &ops_ap);
    // crate kernel_exp
    ArrayPtr kernel_exp(vector<ArrayPtr> &ops_ap);
    // crate kernel_sin
    ArrayPtr kernel_sin(vector<ArrayPtr> &ops_ap);
    // crate kernel_tan
    ArrayPtr kernel_tan(vector<ArrayPtr> &ops_ap);
    // crate kernel_cos
    ArrayPtr kernel_cos(vector<ArrayPtr> &ops_ap);
    // crate kernel_rcp
    ArrayPtr kernel_rcp(vector<ArrayPtr> &ops_ap);
    // crate kernel_sqrt
    ArrayPtr kernel_sqrt(vector<ArrayPtr> &ops_ap);
    // crate kernel_asin
    ArrayPtr kernel_asin(vector<ArrayPtr> &ops_ap);
    // crate kernel_acos
    ArrayPtr kernel_acos(vector<ArrayPtr> &ops_ap);
    // crate kernel_atan
    ArrayPtr kernel_atan(vector<ArrayPtr> &ops_ap);
    // crate kernel_abs
    ArrayPtr kernel_abs(vector<ArrayPtr> &ops_ap);
    // crate kernel_log
    ArrayPtr kernel_log(vector<ArrayPtr> &ops_ap);
    // crate kernel_uplus
    ArrayPtr kernel_uplus(vector<ArrayPtr> &ops_ap);
    // crate kernel_uminus
    ArrayPtr kernel_uminus(vector<ArrayPtr> &ops_ap);
    // crate kernel_log10
    ArrayPtr kernel_log10(vector<ArrayPtr> &ops_ap);
    // crate kernel_tanh
    ArrayPtr kernel_tanh(vector<ArrayPtr> &ops_ap);
    // crate kernel_sinh
    ArrayPtr kernel_sinh(vector<ArrayPtr> &ops_ap);
    // crate kernel_cosh
    ArrayPtr kernel_cosh(vector<ArrayPtr> &ops_ap);
    // crate kernel_or
    ArrayPtr kernel_or(vector<ArrayPtr> &ops_ap);
    // crate kernel_and
    ArrayPtr kernel_and(vector<ArrayPtr> &ops_ap);

    ArrayPtr kernel_pow(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_not(vector<ArrayPtr> &ops_ap);
      
    // ap = u {+ - * /} v
    // A = U + V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_plus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      // support pseudo array calculation
      
      // (1) u is a scalar        
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_plus_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (2) v is a scalar
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_plus_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (3) u's bitset != v's bitset
      } else if (u->get_bitset() != v->get_bitset()) {
        int su = oa::utils::get_shape_dimension(u->local_shape());
        int sv = oa::utils::get_shape_dimension(v->local_shape());

        PartitionPtr pp;
        if (su > sv) pp = u->get_partition();
        else pp = v->get_partition(); 
        
        ap = ArrayPool::global()->get(pp, dt);

        // use pseudo
        if (u->is_pseudo()) {
          if (u->has_pseudo_3d() == false) u->set_pseudo_3d(PSU3D(u));
          u = u->get_pseudo_3d();
        }

        if (v->is_pseudo()) {
          if (v->has_pseudo_3d() == false) v->set_pseudo_3d(PSU3D(v));
          v = v->get_pseudo_3d();
        }

        oa::internal::pseudo_buffer_plus_buffer(
            (T1*) ap->get_buffer(),
            (T2*) u->get_buffer(),
            (T3*) v->get_buffer(),
            ap->get_local_box(),
            u->get_local_box(),
            v->get_local_box(),
            ap->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            pp->get_stencil_width());

      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*          // U and V must have same shape
                    assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_plus_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_plus_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      return ap;
    }

    // A = U - V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_minus(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      // support pseudo array calculation
      
      // (1) u is a scalar        
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_minus_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (2) v is a scalar
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_minus_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (3) u's bitset != v's bitset
      } else if (u->get_bitset() != v->get_bitset()) {
        int su = oa::utils::get_shape_dimension(u->local_shape());
        int sv = oa::utils::get_shape_dimension(v->local_shape());

        PartitionPtr pp;
        if (su > sv) pp = u->get_partition();
        else pp = v->get_partition(); 
        
        ap = ArrayPool::global()->get(pp, dt);

        // use pseudo
        if (u->is_pseudo()) {
          if (u->has_pseudo_3d() == false) u->set_pseudo_3d(PSU3D(u));
          u = u->get_pseudo_3d();
        }

        if (v->is_pseudo()) {
          if (v->has_pseudo_3d() == false) v->set_pseudo_3d(PSU3D(v));
          v = v->get_pseudo_3d();
        }

        oa::internal::pseudo_buffer_minus_buffer(
            (T1*) ap->get_buffer(),
            (T2*) u->get_buffer(),
            (T3*) v->get_buffer(),
            ap->get_local_box(),
            u->get_local_box(),
            v->get_local_box(),
            ap->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            pp->get_stencil_width());

      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*          // U and V must have same shape
                    assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_minus_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_minus_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      return ap;
    }

    // A = U * V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_mult(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      // support pseudo array calculation
      
      // (1) u is a scalar        
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_mult_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (2) v is a scalar
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_mult_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (3) u's bitset != v's bitset
      } else if (u->get_bitset() != v->get_bitset()) {
        int su = oa::utils::get_shape_dimension(u->local_shape());
        int sv = oa::utils::get_shape_dimension(v->local_shape());

        PartitionPtr pp;
        if (su > sv) pp = u->get_partition();
        else pp = v->get_partition(); 
        
        ap = ArrayPool::global()->get(pp, dt);

        // use pseudo
        if (u->is_pseudo()) {
          if (u->has_pseudo_3d() == false) u->set_pseudo_3d(PSU3D(u));
          u = u->get_pseudo_3d();
        }

        if (v->is_pseudo()) {
          if (v->has_pseudo_3d() == false) v->set_pseudo_3d(PSU3D(v));
          v = v->get_pseudo_3d();
        }

        oa::internal::pseudo_buffer_mult_buffer(
            (T1*) ap->get_buffer(),
            (T2*) u->get_buffer(),
            (T3*) v->get_buffer(),
            ap->get_local_box(),
            u->get_local_box(),
            v->get_local_box(),
            ap->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            pp->get_stencil_width());

      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*          // U and V must have same shape
                    assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_mult_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_mult_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      return ap;
    }

    // A = U / V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_divd(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = oa::utils::cast_data_type(u_dt, v_dt);

      // support pseudo array calculation
      
      // (1) u is a scalar        
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_divd_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (2) v is a scalar
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_divd_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (3) u's bitset != v's bitset
      } else if (u->get_bitset() != v->get_bitset()) {
        int su = oa::utils::get_shape_dimension(u->local_shape());
        int sv = oa::utils::get_shape_dimension(v->local_shape());

        PartitionPtr pp;
        if (su > sv) pp = u->get_partition();
        else pp = v->get_partition(); 
        
        ap = ArrayPool::global()->get(pp, dt);

        // use pseudo
        if (u->is_pseudo()) {
          if (u->has_pseudo_3d() == false) u->set_pseudo_3d(PSU3D(u));
          u = u->get_pseudo_3d();
        }

        if (v->is_pseudo()) {
          if (v->has_pseudo_3d() == false) v->set_pseudo_3d(PSU3D(v));
          v = v->get_pseudo_3d();
        }

        oa::internal::pseudo_buffer_divd_buffer(
            (T1*) ap->get_buffer(),
            (T2*) u->get_buffer(),
            (T3*) v->get_buffer(),
            ap->get_local_box(),
            u->get_local_box(),
            v->get_local_box(),
            ap->buffer_shape(),
            u->buffer_shape(),
            v->buffer_shape(),
            pp->get_stencil_width());

      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*          // U and V must have same shape
                    assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_divd_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_divd_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      return ap;
    }




    ///!:mute
    ///!:set K = [['gt','>'], ['ge', '>='], ['lt', '<'],['le', '<='], &
    ///!:['eq','=='], ['ne','/='],['and','&&'],['or','||']]
    ///!:endmute

    // A = U > V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_gt(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_gt_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_gt_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_gt_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_gt_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_gt_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_gt_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_gt_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_gt_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U >= V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_ge(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_ge_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_ge_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_ge_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_ge_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_ge_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_ge_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_ge_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_ge_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U < V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_lt(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_lt_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_lt_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_lt_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_lt_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_lt_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_lt_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_lt_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_lt_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U <= V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_le(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_le_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_le_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_le_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_le_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_le_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_le_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_le_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_le_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U == V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_eq(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_eq_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_eq_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_eq_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_eq_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_eq_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_eq_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_eq_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_eq_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U != V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_ne(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_ne_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_ne_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_ne_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_ne_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_ne_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_ne_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_ne_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_ne_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U || V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_or(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_or_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_or_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_or_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_or_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_or_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_or_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_or_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_or_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

    // A = U && V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_and(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

#ifndef __HAVE_CUDA__
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_and_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_and_const(
            (T1 *) ap->get_buffer(),
            (T2 *) u->get_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_and_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_and_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
#else
      u->memcopy_gpu_to_cpu();
      v->memcopy_gpu_to_cpu();
      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_cpu_buffer();
        oa::internal::const_and_buffer(
            (T1 *) ap->get_cpu_buffer(),
            scalar,
            (T3 *) v->get_cpu_buffer(),
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_cpu_buffer();
        oa::internal::buffer_and_const(
            (T1 *) ap->get_cpu_buffer(),
            (T2 *) u->get_cpu_buffer(),
            scalar,
            //ap->buffer_size()
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width()
                                            );
      } else {
        PartitionPtr upar = u->get_partition();
        PartitionPtr vpar = v->get_partition();
        assert(upar->get_comm() == vpar->get_comm());

        /*        // U and V must have same shape
                  assert(oa::utils::is_equal_shape(upar->shape(), vpar->shape()));
        */
        ap = ArrayPool::global()->get(upar, dt);
        if (upar->equal(vpar)) {
          oa::internal::buffer_and_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) v->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_and_buffer(
              (T1 *) ap->get_cpu_buffer(),
              (T2 *) u->get_cpu_buffer(),
              (T3 *) tmp->get_cpu_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      ap->memcopy_cpu_to_gpu();
#endif
      return ap;
    }

        
  }
}


#endif

    
