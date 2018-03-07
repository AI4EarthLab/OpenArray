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

///:include "../../NodeType.fypp"

namespace oa{
  namespace kernel{

    ///:for t in [i for i in L if i[3] in ['A','B','C','F']]
    ///:set name = t[1]
    ///:set sy = t[2]
    // crate kernel_${name}$
    ArrayPtr kernel_${name}$(vector<ArrayPtr> &ops_ap);
    ///:endfor

    ArrayPtr kernel_pow(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_not(vector<ArrayPtr> &ops_ap);
      
    // ap = u {+ - * /} v
    ///:for k in [i for i in L if i[3] == 'A']
    ///:set name = k[1]
    ///:set sy = k[2]
    // A = U ${sy}$ V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
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
        oa::internal::const_${name}$_buffer(
            (T1 *) ap->get_buffer(),
            scalar,
            (T3 *) v->get_buffer(),
            ap->buffer_shape(),
            ap->get_partition()->get_stencil_width());
      
      // (2) v is a scalar
      } else if (v->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(u->get_partition(), dt);
        T3 scalar = *(T3*) v->get_buffer();
        oa::internal::buffer_${name}$_const(
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

        oa::internal::pseudo_buffer_${name}$_buffer(
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
          oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }

      return ap;
    }

    ///:endfor



    ///!:mute
    ///!:set K = [['gt','>'], ['ge', '>='], ['lt', '<'],['le', '<='], &
    ///!:['eq','=='], ['ne','/='],['and','&&'],['or','||']]
    ///!:endmute

    ///:for t in [i for i in L if i[3] in ['B','F']]
    ///:set name = t[1]
    ///:set sy = t[2]
    // A = U ${sy}$ V
    template <typename T1, typename T2, typename T3>
    ArrayPtr t_kernel_${name}$(vector<ArrayPtr> &ops_ap) {
      ArrayPtr u = ops_ap[0];
      ArrayPtr v = ops_ap[1];
      ArrayPtr ap;

      int u_dt = u->get_data_type();
      int v_dt = v->get_data_type();
      int dt = 0;

      if (u->is_seqs_scalar()) {
        ap = ArrayPool::global()->get(v->get_partition(), dt);
        T2 scalar = *(T2*) u->get_buffer();
        oa::internal::const_${name}$_buffer(
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
        oa::internal::buffer_${name}$_const(
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
          oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) v->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        } else {
          ArrayPtr tmp = oa::funcs::transfer(v, upar);
          oa::internal::buffer_${name}$_buffer(
              (T1 *) ap->get_buffer(),
              (T2 *) u->get_buffer(),
              (T3 *) tmp->get_buffer(),
              //ap->buffer_size()
              ap->buffer_shape(),
              ap->get_partition()->get_stencil_width());
        }
      }
      return ap;
    }

    ///:endfor
        
  }
}


#endif

    
