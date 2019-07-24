#include "config.h"
  
  



























































#include "config.h"
  
  module oa_type
    use iso_c_binding
    !use mpi
    use oa_utils
    type Array
       type(c_ptr) :: ptr = C_NULL_PTR
       integer :: lr = L ! lvalue or rvalue
     contains
       final :: destroy_array
    end type Array

    type Node
       type(c_ptr) :: ptr = C_NULL_PTR
       integer :: lr = L ! lvalue or rvalue
       integer :: id = -1
     contains
       final :: destroy_node
    end type Node

    ! private :: try_destroy, try_destroy_array, try_destroy_node
    ! private :: set_rvalue, set_rvalue_array, set_rvalue_node
    
    interface shape
       module procedure shape_node
       module procedure shape_array
    end interface shape
    
    interface display
       module procedure display_node
       module procedure display_array
    end interface display

    interface disp
       module procedure display_node
       module procedure display_array
    end interface disp

    interface set_rvalue
       module procedure set_rvalue_array
       module procedure set_rvalue_node
    end interface set_rvalue

    interface destroy
       module procedure destroy_node
       module procedure destroy_array
    end interface destroy

    interface try_destroy
       module procedure try_destroy_node
       module procedure try_destroy_array
    end interface try_destroy
    
    interface consts
       module procedure consts_int
       module procedure consts_float
       module procedure consts_double
    end interface consts

    interface local_sub
      module procedure local_sub
!      ///:for t in TYPE
!      module procedure local_sub_double
!      ///:endfor
    end interface local_sub

    interface set_local
      module procedure set_local_int
      module procedure set_local_float
      module procedure set_local_double
    end interface set_local

    ! xiaogang

    interface
       subroutine c_new_seqs_scalar_node_int &
            (ptr, val) &
            bind(C, name = 'c_new_seqs_scalar_node_int')
         use iso_c_binding
         type(c_ptr), intent(inout) :: ptr
         integer, intent(in), VALUE :: val
       end subroutine
    end interface
    interface
       subroutine c_new_seqs_scalar_node_float &
            (ptr, val) &
            bind(C, name = 'c_new_seqs_scalar_node_float')
         use iso_c_binding
         type(c_ptr), intent(inout) :: ptr
         real, intent(in), VALUE :: val
       end subroutine
    end interface
    interface
       subroutine c_new_seqs_scalar_node_double &
            (ptr, val) &
            bind(C, name = 'c_new_seqs_scalar_node_double')
         use iso_c_binding
         type(c_ptr), intent(inout) :: ptr
         real(kind=8), intent(in), VALUE :: val
       end subroutine
    end interface

    interface
       subroutine c_new_seqs_scalar_node_int_simple &
            (val, id) &
            bind(C, name = 'c_new_seqs_scalar_node_int_simple')
         use iso_c_binding
         ! type(c_ptr), intent(inout) :: ptr
         integer, intent(in), VALUE :: val
         integer::id
       end subroutine
    end interface
    interface
       subroutine c_new_seqs_scalar_node_float_simple &
            (val, id) &
            bind(C, name = 'c_new_seqs_scalar_node_float_simple')
         use iso_c_binding
         ! type(c_ptr), intent(inout) :: ptr
         real, intent(in), VALUE :: val
         integer::id
       end subroutine
    end interface
    interface
       subroutine c_new_seqs_scalar_node_double_simple &
            (val, id) &
            bind(C, name = 'c_new_seqs_scalar_node_double_simple')
         use iso_c_binding
         ! type(c_ptr), intent(inout) :: ptr
         real(kind=8), intent(in), VALUE :: val
         integer::id
       end subroutine
    end interface

    !xiaogang
    interface
       subroutine c_new_node_array_simple(B, id) &
            bind(C, name='c_new_node_array_simple')
         use iso_c_binding
         ! type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
        integer, intent(inout)::id
       end subroutine
    end interface

    interface
       subroutine c_new_node_int3_simple_rep(x1, y1, z1, id) &
            bind(C, name='c_new_node_int3_simple_rep')
         use iso_c_binding
         integer :: x1,y1,z1, id
       end subroutine
    end interface

    interface
       subroutine c_new_node_int3_simple_shift(x, op_y, op_z, id) &
            bind(C, name='c_new_node_int3_simple_shift')
         use iso_c_binding
         integer :: x,op_y,op_z, id
       end subroutine
    end interface

    interface
       subroutine c_new_node_op2_simple(nodetype, res_id, id_u, id_v) &
           bind(C, name='c_new_node_op2_simple')
        use iso_c_binding
         integer:: res_id, id_u, id_v
         integer(c_int), intent(in), VALUE :: nodetype
       end subroutine
    end interface
!
    interface
       subroutine c_new_node_max2_simple(res, id1, id2) &
            bind(C, name='c_new_node_max2_simple')
         use iso_c_binding
        implicit none
        integer::res, id1, id2
       end subroutine
    end interface

    interface
       subroutine c_new_node_rep_simple(res, id1, id2) &
            bind(C, name='c_new_node_rep_simple')
            use iso_c_binding
             implicit none
             integer::res, id1, id2
       end subroutine
    end interface

    interface 
        subroutine c_new_node_csum_simple(res, id1, id2) &
            bind(C, name='c_new_node_csum_simple')
         use iso_c_binding
        implicit none
        integer::res, id1, id2
       end subroutine
    end interface
    
    interface
       subroutine c_new_node_sum_simple(res, id1, id2) &
            bind(C, name='c_new_node_sum_simple')
         use iso_c_binding
        implicit none
        integer::res, id1, id2
       end subroutine
    end interface

    interface dxc    
       module procedure dxc_node
       module procedure dxc_array
    end interface dxc    
    interface dyc    
       module procedure dyc_node
       module procedure dyc_array
    end interface dyc    
    interface dzc    
       module procedure dzc_node
       module procedure dzc_array
    end interface dzc    
    interface axb    
       module procedure axb_node
       module procedure axb_array
    end interface axb    
    interface axf    
       module procedure axf_node
       module procedure axf_array
    end interface axf    
    interface ayb    
       module procedure ayb_node
       module procedure ayb_array
    end interface ayb    
    interface ayf    
       module procedure ayf_node
       module procedure ayf_array
    end interface ayf    
    interface azb    
       module procedure azb_node
       module procedure azb_array
    end interface azb    
    interface azf    
       module procedure azf_node
       module procedure azf_array
    end interface azf    
    interface dxb    
       module procedure dxb_node
       module procedure dxb_array
    end interface dxb    
    interface dxf    
       module procedure dxf_node
       module procedure dxf_array
    end interface dxf    
    interface dyb    
       module procedure dyb_node
       module procedure dyb_array
    end interface dyb    
    interface dyf    
       module procedure dyf_node
       module procedure dyf_array
    end interface dyf    
    interface dzb    
       module procedure dzb_node
       module procedure dzb_array
    end interface dzb    
    interface dzf    
       module procedure dzf_node
       module procedure dzf_array
    end interface dzf    

    interface assignment(=)
       module procedure array_assign_array
       module procedure node_assign_node
       module procedure node_assign_array
    end interface assignment(=)

    interface
       subroutine c_array_assign_array(A, B, pa, pb) &
            bind(C, name='c_array_assign_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
         integer(c_int), intent(inout) :: pa
         integer(c_int), intent(in) :: pb
       end subroutine
    end interface

    interface
       subroutine c_node_assign_node(A, B) &
            bind(C, name='c_node_assign_node')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
       end subroutine
    end interface

    interface
       subroutine c_node_assign_array(A) &
            bind(C, name='c_node_assign_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
       end subroutine
    end interface

    interface operator (+)
       module procedure ops_double_plus_array
       module procedure ops_double_plus_node
       module procedure ops_float_plus_array
       module procedure ops_float_plus_node
       module procedure ops_int_plus_array
       module procedure ops_int_plus_node
       module procedure ops_array_plus_double
       module procedure ops_array_plus_float
       module procedure ops_array_plus_int
       module procedure ops_array_plus_array
       module procedure ops_array_plus_node
       module procedure ops_node_plus_double
       module procedure ops_node_plus_float
       module procedure ops_node_plus_int
       module procedure ops_node_plus_array
       module procedure ops_node_plus_node
    end interface operator (+)
    interface operator (-)
       module procedure ops_double_minus_array
       module procedure ops_double_minus_node
       module procedure ops_float_minus_array
       module procedure ops_float_minus_node
       module procedure ops_int_minus_array
       module procedure ops_int_minus_node
       module procedure ops_array_minus_double
       module procedure ops_array_minus_float
       module procedure ops_array_minus_int
       module procedure ops_array_minus_array
       module procedure ops_array_minus_node
       module procedure ops_node_minus_double
       module procedure ops_node_minus_float
       module procedure ops_node_minus_int
       module procedure ops_node_minus_array
       module procedure ops_node_minus_node
    end interface operator (-)
    interface operator (*)
       module procedure ops_double_mult_array
       module procedure ops_double_mult_node
       module procedure ops_float_mult_array
       module procedure ops_float_mult_node
       module procedure ops_int_mult_array
       module procedure ops_int_mult_node
       module procedure ops_array_mult_double
       module procedure ops_array_mult_float
       module procedure ops_array_mult_int
       module procedure ops_array_mult_array
       module procedure ops_array_mult_node
       module procedure ops_node_mult_double
       module procedure ops_node_mult_float
       module procedure ops_node_mult_int
       module procedure ops_node_mult_array
       module procedure ops_node_mult_node
    end interface operator (*)
    interface operator (/)
       module procedure ops_double_divd_array
       module procedure ops_double_divd_node
       module procedure ops_float_divd_array
       module procedure ops_float_divd_node
       module procedure ops_int_divd_array
       module procedure ops_int_divd_node
       module procedure ops_array_divd_double
       module procedure ops_array_divd_float
       module procedure ops_array_divd_int
       module procedure ops_array_divd_array
       module procedure ops_array_divd_node
       module procedure ops_node_divd_double
       module procedure ops_node_divd_float
       module procedure ops_node_divd_int
       module procedure ops_node_divd_array
       module procedure ops_node_divd_node
    end interface operator (/)
    interface operator (>)
       module procedure ops_double_gt_array
       module procedure ops_double_gt_node
       module procedure ops_float_gt_array
       module procedure ops_float_gt_node
       module procedure ops_int_gt_array
       module procedure ops_int_gt_node
       module procedure ops_array_gt_double
       module procedure ops_array_gt_float
       module procedure ops_array_gt_int
       module procedure ops_array_gt_array
       module procedure ops_array_gt_node
       module procedure ops_node_gt_double
       module procedure ops_node_gt_float
       module procedure ops_node_gt_int
       module procedure ops_node_gt_array
       module procedure ops_node_gt_node
    end interface operator (>)
    interface operator (>=)
       module procedure ops_double_ge_array
       module procedure ops_double_ge_node
       module procedure ops_float_ge_array
       module procedure ops_float_ge_node
       module procedure ops_int_ge_array
       module procedure ops_int_ge_node
       module procedure ops_array_ge_double
       module procedure ops_array_ge_float
       module procedure ops_array_ge_int
       module procedure ops_array_ge_array
       module procedure ops_array_ge_node
       module procedure ops_node_ge_double
       module procedure ops_node_ge_float
       module procedure ops_node_ge_int
       module procedure ops_node_ge_array
       module procedure ops_node_ge_node
    end interface operator (>=)
    interface operator (<)
       module procedure ops_double_lt_array
       module procedure ops_double_lt_node
       module procedure ops_float_lt_array
       module procedure ops_float_lt_node
       module procedure ops_int_lt_array
       module procedure ops_int_lt_node
       module procedure ops_array_lt_double
       module procedure ops_array_lt_float
       module procedure ops_array_lt_int
       module procedure ops_array_lt_array
       module procedure ops_array_lt_node
       module procedure ops_node_lt_double
       module procedure ops_node_lt_float
       module procedure ops_node_lt_int
       module procedure ops_node_lt_array
       module procedure ops_node_lt_node
    end interface operator (<)
    interface operator (<=)
       module procedure ops_double_le_array
       module procedure ops_double_le_node
       module procedure ops_float_le_array
       module procedure ops_float_le_node
       module procedure ops_int_le_array
       module procedure ops_int_le_node
       module procedure ops_array_le_double
       module procedure ops_array_le_float
       module procedure ops_array_le_int
       module procedure ops_array_le_array
       module procedure ops_array_le_node
       module procedure ops_node_le_double
       module procedure ops_node_le_float
       module procedure ops_node_le_int
       module procedure ops_node_le_array
       module procedure ops_node_le_node
    end interface operator (<=)
    interface operator (==)
       module procedure ops_double_eq_array
       module procedure ops_double_eq_node
       module procedure ops_float_eq_array
       module procedure ops_float_eq_node
       module procedure ops_int_eq_array
       module procedure ops_int_eq_node
       module procedure ops_array_eq_double
       module procedure ops_array_eq_float
       module procedure ops_array_eq_int
       module procedure ops_array_eq_array
       module procedure ops_array_eq_node
       module procedure ops_node_eq_double
       module procedure ops_node_eq_float
       module procedure ops_node_eq_int
       module procedure ops_node_eq_array
       module procedure ops_node_eq_node
    end interface operator (==)
    interface operator (/=)
       module procedure ops_double_ne_array
       module procedure ops_double_ne_node
       module procedure ops_float_ne_array
       module procedure ops_float_ne_node
       module procedure ops_int_ne_array
       module procedure ops_int_ne_node
       module procedure ops_array_ne_double
       module procedure ops_array_ne_float
       module procedure ops_array_ne_int
       module procedure ops_array_ne_array
       module procedure ops_array_ne_node
       module procedure ops_node_ne_double
       module procedure ops_node_ne_float
       module procedure ops_node_ne_int
       module procedure ops_node_ne_array
       module procedure ops_node_ne_node
    end interface operator (/=)
    interface operator (.or.)
       module procedure ops_double_or_array
       module procedure ops_double_or_node
       module procedure ops_float_or_array
       module procedure ops_float_or_node
       module procedure ops_int_or_array
       module procedure ops_int_or_node
       module procedure ops_array_or_double
       module procedure ops_array_or_float
       module procedure ops_array_or_int
       module procedure ops_array_or_array
       module procedure ops_array_or_node
       module procedure ops_node_or_double
       module procedure ops_node_or_float
       module procedure ops_node_or_int
       module procedure ops_node_or_array
       module procedure ops_node_or_node
    end interface operator (.or.)
    interface operator (.and.)
       module procedure ops_double_and_array
       module procedure ops_double_and_node
       module procedure ops_float_and_array
       module procedure ops_float_and_node
       module procedure ops_int_and_array
       module procedure ops_int_and_node
       module procedure ops_array_and_double
       module procedure ops_array_and_float
       module procedure ops_array_and_int
       module procedure ops_array_and_array
       module procedure ops_array_and_node
       module procedure ops_node_and_double
       module procedure ops_node_and_float
       module procedure ops_node_and_int
       module procedure ops_node_and_array
       module procedure ops_node_and_node
    end interface operator (.and.)

    interface exp 
       module procedure ops_exp_array
       module procedure ops_exp_node
    end interface exp 
    interface sin 
       module procedure ops_sin_array
       module procedure ops_sin_node
    end interface sin 
    interface tan 
       module procedure ops_tan_array
       module procedure ops_tan_node
    end interface tan 
    interface cos 
       module procedure ops_cos_array
       module procedure ops_cos_node
    end interface cos 
    interface rcp 
       module procedure ops_rcp_array
       module procedure ops_rcp_node
    end interface rcp 
    interface sqrt 
       module procedure ops_sqrt_array
       module procedure ops_sqrt_node
    end interface sqrt 
    interface asin 
       module procedure ops_asin_array
       module procedure ops_asin_node
    end interface asin 
    interface acos 
       module procedure ops_acos_array
       module procedure ops_acos_node
    end interface acos 
    interface atan 
       module procedure ops_atan_array
       module procedure ops_atan_node
    end interface atan 
    interface abs 
       module procedure ops_abs_array
       module procedure ops_abs_node
    end interface abs 
    interface log 
       module procedure ops_log_array
       module procedure ops_log_node
    end interface log 
    interface operator (+) 
       module procedure ops_uplus_array
       module procedure ops_uplus_node
    end interface operator (+) 
    interface operator (-) 
       module procedure ops_uminus_array
       module procedure ops_uminus_node
    end interface operator (-) 
    interface log10 
       module procedure ops_log10_array
       module procedure ops_log10_node
    end interface log10 
    interface tanh 
       module procedure ops_tanh_array
       module procedure ops_tanh_node
    end interface tanh 
    interface sinh 
       module procedure ops_sinh_array
       module procedure ops_sinh_node
    end interface sinh 
    interface cosh 
       module procedure ops_cosh_array
       module procedure ops_cosh_node
    end interface cosh 

    interface operator(**)
       module procedure ops_pow_node_int
       module procedure ops_pow_node_float
       module procedure ops_pow_node_double
       module procedure ops_pow_array_int
       module procedure ops_pow_array_float
       module procedure ops_pow_array_double
    end interface operator(**)
    
    interface get_local_buffer
       module procedure get_local_buffer_int
       module procedure get_local_buffer_float
       module procedure get_local_buffer_double
       module procedure get_local_buffer_int_with_location
       module procedure get_local_buffer_float_with_location
       module procedure get_local_buffer_double_with_location
    end interface get_local_buffer

    interface is_rvalue
       module procedure is_rvalue_node
       module procedure is_rvalue_array
    end interface is_rvalue
    
    integer, parameter :: OA_INT    = 0
    integer, parameter :: OA_FLOAT  = 1
    integer, parameter :: OA_DOUBLE = 2

    integer, parameter :: STENCIL_STAR = 0
    integer, parameter :: STENCIL_BOX  = 1
    
    integer :: default_data_type     = OA_FLOAT
    integer :: default_stencil_type  = STENCIL_BOX    
    integer :: default_stencil_width = 1

  contains

    subroutine format_short()
      implicit none
      interface
         subroutine c_set_disp_format(f) &
              bind(C, name = 'c_format_short')
              use iso_c_binding
              integer(c_int), value :: f
         end subroutine
      end interface

      call c_set_disp_format(0)
    end subroutine

    subroutine format_long()
      implicit none
      interface
         subroutine c_set_disp_format(f) &
              bind(C, name = 'c_format_short')
              use iso_c_binding
              integer(c_int), value :: f
         end subroutine
      end interface

      call c_set_disp_format(1)
    end subroutine
    
    function is_rvalue_array(A) result(res)
      implicit none
      type(Array), intent(in) :: A
      logical :: res
      
      res = (A%lr .eq. RVALUE)
    end function

    function is_rvalue_node(A) result(res)
      implicit none
      type(Node), intent(in) :: A
      logical :: res
      
      res = (A%lr .eq. RVALUE)
    end function

    !> destroy the associated shared_ptr in C++
    !> intent(in) is used for trick
    subroutine destroy_array(A)
      use iso_c_binding
      type(Array), intent(in) :: A

      interface
         subroutine c_destroy_array(A) &
              bind(C, name = 'c_destroy_array')
           use iso_c_binding
           type(c_ptr), intent(in) :: A
         end subroutine
      end interface

      call c_destroy_array(A%ptr)

    end subroutine

    !> destroy the associated shared_ptr in C++
    !> intent(in) is used for trick.
    !> this subroutine is used ONLY for local variables
    subroutine destroy_node(A)
      use iso_c_binding 
      type(Node), intent(in) :: A

      interface
         subroutine c_destroy_node(A) &
              bind(C, name = 'c_destroy_node')
           use iso_c_binding
           type(c_ptr), intent(in) :: A
         end subroutine
      end interface

      call c_destroy_node(A%ptr)
      
    end subroutine

    !> destroy the object if it is 'rvalue'
    !> this subroutine is used to destroy the parameter
    subroutine try_destroy_array(A)
      use iso_c_binding
      type(Array), intent(in) :: A

      if(is_rvalue(A)) call destroy(A)
      
    end subroutine

    !> destroy the object if it is 'rvalue'
    !> this subroutine is used to destroy the parameter
    subroutine try_destroy_node(A)
      use iso_c_binding
      type(Node), intent(in) :: A

      if(is_rvalue(A)) call destroy(A)
      
    end subroutine
    
    subroutine display_info(A, prefix)
      use iso_c_binding
      type(array), intent(in) :: A
      character(len=*):: prefix
      interface
         subroutine c_display_array_info(A, prefix) &
              bind(C, name = 'c_display_array_info')
           use iso_c_binding
           type(c_ptr), intent(in), VALUE :: A
           character(kind=c_char),  intent(in)  :: prefix(*)
         end subroutine
      end interface

      !ASSERT_LVALUE(A)      
      call c_display_array_info(A%ptr, string_f2c(prefix))
    end subroutine
    
    subroutine display_array(A, prefix)
      use iso_c_binding
      type(array), intent(in) :: A
      character(len=*):: prefix
      interface
         subroutine c_display_array(A, prefix) &
              bind(C, name = 'c_display_array')
           use iso_c_binding
           type(c_ptr), intent(in), VALUE :: A
           character(kind=c_char),  intent(in)  :: prefix(*)
         end subroutine
      end interface

      !ASSERT_LVALUE(A)      
      call c_display_array(A%ptr, string_f2c(prefix))
    end subroutine
    
    subroutine display_node(A, prefix)
      use iso_c_binding
      type(node), intent(in) :: A
      character(len=*) :: prefix

      interface
         subroutine c_display_node(A, prefix) &
              bind(C, name = 'c_display_node')
           use iso_c_binding
           type(c_ptr), intent(in), VALUE :: A
           character(kind=c_char),  intent(in)  :: prefix(*)
         end subroutine
      end interface

      call c_display_node(A%ptr, string_f2c(prefix))
    end subroutine

    function ones(m, n, k, sw, dt) result(A)
      integer(c_int) :: m, op_n, op_k, op_sw, op_dt
      integer(c_int), optional :: n, k, sw, dt
      type(Array) :: A

      interface
         subroutine c_ones(ptr, m, n, k, &
              op_sw, op_dt) &
              bind(C, name = 'c_ones')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw, op_dt
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      if (present(dt)) then
         op_dt = dt
      else
         op_dt = default_data_type
      endif

      if(present(n)) then
         op_n = n
      else
         op_n = 1
      end if

      if(present(k)) then
         op_k = k
      else
         op_k = 1
      end if
      
      call c_ones(A%ptr, m, op_n, op_k, op_sw, op_dt)
      
      call set_rvalue(A)
      
    end function
    function zeros(m, n, k, sw, dt) result(A)
      integer(c_int) :: m, op_n, op_k, op_sw, op_dt
      integer(c_int), optional :: n, k, sw, dt
      type(Array) :: A

      interface
         subroutine c_zeros(ptr, m, n, k, &
              op_sw, op_dt) &
              bind(C, name = 'c_zeros')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw, op_dt
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      if (present(dt)) then
         op_dt = dt
      else
         op_dt = default_data_type
      endif

      if(present(n)) then
         op_n = n
      else
         op_n = 1
      end if

      if(present(k)) then
         op_k = k
      else
         op_k = 1
      end if
      
      call c_zeros(A%ptr, m, op_n, op_k, op_sw, op_dt)
      
      call set_rvalue(A)
      
    end function
    function rands(m, n, k, sw, dt) result(A)
      integer(c_int) :: m, op_n, op_k, op_sw, op_dt
      integer(c_int), optional :: n, k, sw, dt
      type(Array) :: A

      interface
         subroutine c_rands(ptr, m, n, k, &
              op_sw, op_dt) &
              bind(C, name = 'c_rands')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw, op_dt
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      if (present(dt)) then
         op_dt = dt
      else
         op_dt = default_data_type
      endif

      if(present(n)) then
         op_n = n
      else
         op_n = 1
      end if

      if(present(k)) then
         op_k = k
      else
         op_k = 1
      end if
      
      call c_rands(A%ptr, m, op_n, op_k, op_sw, op_dt)
      
      call set_rvalue(A)
      
    end function
    function seqs(m, n, k, sw, dt) result(A)
      integer(c_int) :: m, op_n, op_k, op_sw, op_dt
      integer(c_int), optional :: n, k, sw, dt
      type(Array) :: A

      interface
         subroutine c_seqs(ptr, m, n, k, &
              op_sw, op_dt) &
              bind(C, name = 'c_seqs')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw, op_dt
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      if (present(dt)) then
         op_dt = dt
      else
         op_dt = default_data_type
      endif

      if(present(n)) then
         op_n = n
      else
         op_n = 1
      end if

      if(present(k)) then
         op_k = k
      else
         op_k = 1
      end if
      
      call c_seqs(A%ptr, m, op_n, op_k, op_sw, op_dt)
      
      call set_rvalue(A)
      
    end function

    function consts_int(m, n, k, val, sw) result(A)
      integer(c_int) :: m, n, k, op_sw
      integer(c_int), optional :: sw
      integer :: val
      type(Array) :: A

      interface
         subroutine c_consts_int &
              (ptr, m, n, k, val, op_sw) &
              bind(C, name='c_consts_int')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw
           integer, intent(in), VALUE :: val
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      call c_consts_int(A%ptr, &
           m, n, k, val, op_sw)
      call set_rvalue(A)
    end function
    function consts_float(m, n, k, val, sw) result(A)
      integer(c_int) :: m, n, k, op_sw
      integer(c_int), optional :: sw
      real :: val
      type(Array) :: A

      interface
         subroutine c_consts_float &
              (ptr, m, n, k, val, op_sw) &
              bind(C, name='c_consts_float')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw
           real, intent(in), VALUE :: val
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      call c_consts_float(A%ptr, &
           m, n, k, val, op_sw)
      call set_rvalue(A)
    end function
    function consts_double(m, n, k, val, sw) result(A)
      integer(c_int) :: m, n, k, op_sw
      integer(c_int), optional :: sw
      real(kind=8) :: val
      type(Array) :: A

      interface
         subroutine c_consts_double &
              (ptr, m, n, k, val, op_sw) &
              bind(C, name='c_consts_double')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw
           real(kind=8), intent(in), VALUE :: val
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      call c_consts_double(A%ptr, &
           m, n, k, val, op_sw)
      call set_rvalue(A)
    end function

    function new_local_int3(v) result(A)
      implicit none
      integer :: v(3)
      type(node) :: A

      interface
         subroutine c_new_local_int3(A, v) &
              bind(C, name='c_new_local_int3')
           use iso_c_binding
           implicit none
           type(c_ptr) :: A
           integer :: v(3)
         end subroutine
      end interface

      call c_new_local_int3(A%ptr, v)

      call set_rvalue(A)
    end function


    subroutine grid_init(ch, A, B, C)
      use iso_c_binding
      character(len=1) :: ch
      type(Array), intent(in) :: A, B, C

      interface
         subroutine c_grid_init(ch, A, B, C) &
              bind(C, name = 'c_grid_init')
           use iso_c_binding
           character(len=1) :: ch
           type(c_ptr), intent(in) :: A, B, C
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      !ASSERT_LVALUE(B)
      !ASSERT_LVALUE(C)
      
      call c_grid_init(ch, A%ptr, B%ptr, C%ptr)

    end subroutine

    subroutine grid_bind(A, pos)
      use iso_c_binding
      type(Array), intent(in) :: A
      integer :: pos
      interface
         subroutine c_grid_bind(A, pos) &
              bind(C, name = 'c_grid_bind')
           use iso_c_binding
           type(c_ptr), intent(in) :: A
           integer , value :: pos
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      
      call c_grid_bind(A%ptr, pos)

    end subroutine


    function dxc_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxc_simple(id1, id2) &
              bind(C, name='c_new_node_dxc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dxc_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyc_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyc_simple(id1, id2) &
              bind(C, name='c_new_node_dyc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dyc_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzc_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzc_simple(id1, id2) &
              bind(C, name='c_new_node_dzc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dzc_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function axb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_axb_simple(id1, id2) &
              bind(C, name='c_new_node_axb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_axb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function axf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_axf_simple(id1, id2) &
              bind(C, name='c_new_node_axf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_axf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function ayb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_ayb_simple(id1, id2) &
              bind(C, name='c_new_node_ayb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_ayb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function ayf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_ayf_simple(id1, id2) &
              bind(C, name='c_new_node_ayf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_ayf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function azb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_azb_simple(id1, id2) &
              bind(C, name='c_new_node_azb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_azb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function azf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_azf_simple(id1, id2) &
              bind(C, name='c_new_node_azf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_azf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dxb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxb_simple(id1, id2) &
              bind(C, name='c_new_node_dxb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dxb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dxf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxf_simple(id1, id2) &
              bind(C, name='c_new_node_dxf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dxf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyb_simple(id1, id2) &
              bind(C, name='c_new_node_dyb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dyb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyf_simple(id1, id2) &
              bind(C, name='c_new_node_dyf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dyf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzb_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzb_simple(id1, id2) &
              bind(C, name='c_new_node_dzb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dzb_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzf_node(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzf_simple(id1, id2) &
              bind(C, name='c_new_node_dzf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(node) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_dzf_simple(B%id, A%id)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dxc_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxc_simple(id1, id2) &
              bind(C, name='c_new_node_dxc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dxc_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyc_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyc_simple(id1, id2) &
              bind(C, name='c_new_node_dyc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dyc_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzc_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzc_simple(id1, id2) &
              bind(C, name='c_new_node_dzc_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dzc_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function axb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_axb_simple(id1, id2) &
              bind(C, name='c_new_node_axb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_axb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function axf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_axf_simple(id1, id2) &
              bind(C, name='c_new_node_axf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_axf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function ayb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_ayb_simple(id1, id2) &
              bind(C, name='c_new_node_ayb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_ayb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function ayf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_ayf_simple(id1, id2) &
              bind(C, name='c_new_node_ayf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_ayf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function azb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_azb_simple(id1, id2) &
              bind(C, name='c_new_node_azb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_azb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function azf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_azf_simple(id1, id2) &
              bind(C, name='c_new_node_azf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_azf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dxb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxb_simple(id1, id2) &
              bind(C, name='c_new_node_dxb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dxb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dxf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dxf_simple(id1, id2) &
              bind(C, name='c_new_node_dxf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dxf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyb_simple(id1, id2) &
              bind(C, name='c_new_node_dyb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dyb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dyf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dyf_simple(id1, id2) &
              bind(C, name='c_new_node_dyf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dyf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzb_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzb_simple(id1, id2) &
              bind(C, name='c_new_node_dzb_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dzb_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function
    function dzf_array(A) result(B)
      implicit none

      interface
         subroutine c_new_node_dzf_simple(id1, id2) &
              bind(C, name='c_new_node_dzf_simple')
           use iso_c_binding
           ! type(c_ptr), intent(inout) :: A
           ! type(c_ptr), intent(in) :: U 
           integer :: id1, id2
         end subroutine
      end interface

      type(array) :: A
      type(node)  :: B
      type(node) :: NA
      integer:: id_a

      call c_new_node_array_simple(A%ptr, id_a)
      
      call c_new_node_dzf_simple(B%id, id_a)
      !!B%ptr = C_NULL_PTR

      ! call set_rvalue(B)
      
      ! call try_destroy(A)
      ! call destroy(NA)
    end function

    function local_shape(A) result(res)
      implicit none
      interface
         subroutine c_local_shape(A, res) &
              bind(C, name = "c_local_shape")
           use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer, intent(out) :: res(3)
         end subroutine
      end interface

      type(array) :: A
      integer :: res(3)

      !ASSERT_LVALUE(A)
      
      call c_local_shape(A%ptr, res)
    end function

    function buffer_shape(A) result(res)
      implicit none
      interface
         subroutine c_buffer_shape(A, res) &
              bind(C, name = "c_buffer_shape")
           use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer, intent(out) :: res(3)
         end subroutine
      end interface

      type(array) :: A
      integer :: res(3)

      !ASSERT_LVALUE(A)
      
      call c_buffer_shape(A%ptr, res)
    end function

    function get_buffer_ptr(A) result(res)
      implicit none
      interface
         subroutine c_get_buffer_ptr(A, res) &
              bind(C, name = "c_get_buffer_ptr")
           use iso_c_binding
           implicit none
           type(c_ptr) :: a
           type(c_ptr), intent(out) :: res
         end subroutine
      end interface

      type(array) :: A
      type(c_ptr) :: res

      !ASSERT_LVALUE(A)
      
      call c_get_buffer_ptr(A%ptr, res)
      
    end function

    subroutine get_local_buffer_int(res, A)
      use iso_c_binding
      implicit none
      type(array) :: A
      integer, dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: flag
      interface
         subroutine c_is_array_int(flag, A) &
              bind(C, name = "c_is_array_int")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_int (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    subroutine get_local_buffer_float(res, A)
      use iso_c_binding
      implicit none
      type(array) :: A
      real, dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: flag
      interface
         subroutine c_is_array_float(flag, A) &
              bind(C, name = "c_is_array_float")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_float (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    subroutine get_local_buffer_double(res, A)
      use iso_c_binding
      implicit none
      type(array) :: A
      real(8), dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: flag
      interface
         subroutine c_is_array_double(flag, A) &
              bind(C, name = "c_is_array_double")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_double (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    
    subroutine get_local_buffer_int_with_location(res, A, sl)
      use iso_c_binding
      implicit none
      type(array) :: A
      integer, dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: sl(6)
      integer :: flag
      interface
         subroutine c_is_array_int(flag, A) &
              bind(C, name = "c_is_array_int")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface
      interface
         subroutine get_global_location(A,is,ie,js,je,ks,ke) &
              bind(C, name = "c_get_global_location")
           use iso_c_binding
           type(c_ptr) :: A
           integer :: is,ie,js,je,ks,ke
         end subroutine
      end interface
      !ASSERT_LVALUE(A)
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_int (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
         call get_global_location(A%ptr,sl(1),sl(2),sl(3),sl(4),sl(5),sl(6))
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    subroutine get_local_buffer_float_with_location(res, A, sl)
      use iso_c_binding
      implicit none
      type(array) :: A
      real, dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: sl(6)
      integer :: flag
      interface
         subroutine c_is_array_float(flag, A) &
              bind(C, name = "c_is_array_float")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface
      interface
         subroutine get_global_location(A,is,ie,js,je,ks,ke) &
              bind(C, name = "c_get_global_location")
           use iso_c_binding
           type(c_ptr) :: A
           integer :: is,ie,js,je,ks,ke
         end subroutine
      end interface
      !ASSERT_LVALUE(A)
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_float (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
         call get_global_location(A%ptr,sl(1),sl(2),sl(3),sl(4),sl(5),sl(6))
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    subroutine get_local_buffer_double_with_location(res, A, sl)
      use iso_c_binding
      implicit none
      type(array) :: A
      real(8), dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: sl(6)
      integer :: flag
      interface
         subroutine c_is_array_double(flag, A) &
              bind(C, name = "c_is_array_double")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface
      interface
         subroutine get_global_location(A,is,ie,js,je,ks,ke) &
              bind(C, name = "c_get_global_location")
           use iso_c_binding
           type(c_ptr) :: A
           integer :: is,ie,js,je,ks,ke
         end subroutine
      end interface
      !ASSERT_LVALUE(A)
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_double (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
         call get_global_location(A%ptr,sl(1),sl(2),sl(3),sl(4),sl(5),sl(6))
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine 
    function shape_array(A) result(res)
      implicit none
      interface
         subroutine c_shape_array(a, s) &
              bind(C, name = "c_shape_array")
           use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer, intent(out) :: s(3)
         end subroutine
      end interface

      type(array) :: A
      integer :: res(3)

      !ASSERT_LVALUE(A)
      
      call c_shape_array(A%ptr, res)

    end function

    function get_box_corners(A) result(res)
      implicit none
      interface
        subroutine c_get_box_corners(a, s) &
          bind(C, name = "c_get_box_corners")
        use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer, intent(out) :: s(6)
         end subroutine
      end interface

      type(array) :: A
      integer :: res(6)

      call c_get_box_corners(A%ptr, res)

    end function

    subroutine update_ghost(A)
      implicit none
      interface
        subroutine c_update_ghost(a) &
          bind(C, name = "c_update_ghost")
        use iso_c_binding
           implicit none
           type(c_ptr) :: a
         end subroutine
      end interface

      type(array) :: A
      call c_update_ghost(A%ptr)
    end subroutine 

    function local_sub(A, x, y, z) result(res)
      implicit none
      interface
        subroutine c_local_sub(a, x, y, z, s) &
          bind(C, name = "c_local_sub")
        use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           real(8), intent(out) :: s
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      real(8) :: res
      rx = x - 1
      ry = y - 1
      rz = z - 1

      call c_local_sub(A%ptr, rx, ry, rz, res)

    end function
!    ///:for t in TYPE
!    subroutine local_sub_double(A, x, y, z, val)
!      implicit none
!      interface
!        subroutine c_local_sub_double(a, x, y, z, s) &
!          bind(C, name = "c_local_sub_double")
!        use iso_c_binding
!           implicit none
!           type(c_ptr) :: a
!           integer :: x, y, z
!           real(8), intent(out) :: s
!         end subroutine
!      end interface
!
!      type(array) :: A
!      integer :: x, y, z
!      integer :: rx, ry, rz
!      real(8) :: val
!      rx = x - 1
!      ry = y - 1
!      rz = z - 1
!
!      call c_local_sub_double(A%ptr, rx, ry, rz, val)
!
!    end subroutine
!
!    ///:endfor

    subroutine set_local_int(A, x, y, z, val)
      implicit none
      interface
        subroutine c_set_local_int(a, x, y, z, val) &
          bind(C, name = "c_set_local_int") 
        use iso_c_binding 
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           integer :: val
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      integer :: val
      rx = x - 1 
      ry = y - 1
      rz = z - 1

      call c_set_local_int(A%ptr, rx, ry, rz, val)

    end subroutine 
    
    subroutine set_local_float(A, x, y, z, val)
      implicit none
      interface
        subroutine c_set_local_float(a, x, y, z, val) &
          bind(C, name = "c_set_local_float") 
        use iso_c_binding 
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           real :: val
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      real :: val
      rx = x - 1 
      ry = y - 1
      rz = z - 1

      call c_set_local_float(A%ptr, rx, ry, rz, val)

    end subroutine 
    
    subroutine set_local_double(A, x, y, z, val)
      implicit none
      interface
        subroutine c_set_local_double(a, x, y, z, val) &
          bind(C, name = "c_set_local_double") 
        use iso_c_binding 
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           real(kind=8) :: val
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      real(kind=8) :: val
      rx = x - 1 
      ry = y - 1
      rz = z - 1

      call c_set_local_double(A%ptr, rx, ry, rz, val)

    end subroutine 
    

    function shape_node(A) result(res)
      implicit none
      interface
         subroutine c_shape_node(a, s) &
              bind(C, name = "c_shape_node")
           use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer, intent(out) :: s(3)
         end subroutine
      end interface

      type(node) :: A
      integer :: res(3)

      !ASSERT_LVALUE(A)
      
      call c_shape_node(A%ptr, res)
      
    end function


    function ops_double_plus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_plus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_plus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_plus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_plus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_plus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_plus_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_plus_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_plus_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_plus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_plus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_plus_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_plus_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_plus_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_plus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_plus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_plus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_plus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_plus_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_minus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_minus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_minus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_minus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_minus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_minus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_minus_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_minus_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_minus_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_minus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_minus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_minus_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_minus_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_minus_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_minus_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_minus_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_minus_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_minus_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_minus_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_mult_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_mult_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_mult_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_mult_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_mult_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_mult_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_mult_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_mult_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_mult_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_mult_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_mult_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_mult_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_mult_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_mult_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_mult_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_mult_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_mult_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_mult_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_mult_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_divd_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_divd_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_divd_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_divd_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_divd_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_divd_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_divd_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_divd_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_divd_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_divd_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_divd_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_divd_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_divd_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_divd_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_divd_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_divd_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_divd_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_divd_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_divd_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_gt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_gt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_gt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_gt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_gt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_gt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_gt_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_gt_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_gt_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_gt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_gt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_gt_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_gt_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_gt_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_gt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_gt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_gt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_gt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_gt_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_ge_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_ge_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_ge_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_ge_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_ge_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_ge_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ge_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ge_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ge_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ge_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ge_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ge_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ge_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ge_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ge_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ge_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ge_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ge_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_ge_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_lt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_lt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_lt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_lt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_lt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_lt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_lt_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_lt_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_lt_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_lt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_lt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_lt_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_lt_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_lt_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_lt_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_lt_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_lt_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_lt_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_lt_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_le_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_le_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_le_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_le_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_le_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_le_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_le_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_le_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_le_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_le_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_le_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_le_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_le_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_le_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_le_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_le_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_le_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_le_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_le_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_eq_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_eq_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_eq_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_eq_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_eq_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_eq_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_eq_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_eq_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_eq_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_eq_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_eq_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_eq_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_eq_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_eq_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_eq_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_eq_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_eq_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_eq_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_eq_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_ne_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_ne_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_ne_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_ne_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_ne_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_ne_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ne_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ne_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ne_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ne_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_ne_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ne_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ne_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ne_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ne_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_ne_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_ne_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_ne_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_ne_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_or_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_or_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_or_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_or_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_or_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_or_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_or_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_or_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_or_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_or_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_or_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_or_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_or_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_or_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_or_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_or_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_or_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_or_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_or_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_and_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_double_and_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real(8), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_double_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_and_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_float_and_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      real, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_float_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_and_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_int_and_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      integer, intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_seqs_scalar_node_int_simple(A, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_and_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_and_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_and_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_and_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_array_and_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(array), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      call c_new_node_array_simple(A%ptr, id_a)

      id_b = B % id
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_and_double(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_and_float(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_and_int(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_and_array(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(array), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      call c_new_node_array_simple(B%ptr, id_b) ! xiaogang
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function

    function ops_node_and_node(A, B) &
         result(res)
      implicit none
      ! xiaogang
      interface
         subroutine c_new_node_and_simple(id_res, id1, id2) &
              bind(C, name='c_new_node_and_simple')
           use iso_c_binding
           implicit none
          integer :: id_res, id1, id2
         end subroutine
      end interface
      type(node), intent(in) :: A
      type(node), intent(in) :: B
      type(node) :: res
      integer::id_a, id_b

      id_a = A%id

      id_b = B % id
      ! xiaogang
      call c_new_node_and_simple(res%id, id_a, id_b)
            
    end function



    function ops_exp_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_exp_simple(id_res, id_a) &
              bind(C, name='c_new_node_exp_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_exp_simple(res%id, id_a)

    end function

    function ops_exp_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_exp_simple(id_res, id_a) &
              bind(C, name='c_new_node_exp_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_exp_simple(res%id, A%id)

    end function

    function ops_sin_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sin_simple(id_res, id_a) &
              bind(C, name='c_new_node_sin_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_sin_simple(res%id, id_a)

    end function

    function ops_sin_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sin_simple(id_res, id_a) &
              bind(C, name='c_new_node_sin_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_sin_simple(res%id, A%id)

    end function

    function ops_tan_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_tan_simple(id_res, id_a) &
              bind(C, name='c_new_node_tan_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_tan_simple(res%id, id_a)

    end function

    function ops_tan_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_tan_simple(id_res, id_a) &
              bind(C, name='c_new_node_tan_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_tan_simple(res%id, A%id)

    end function

    function ops_cos_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_cos_simple(id_res, id_a) &
              bind(C, name='c_new_node_cos_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_cos_simple(res%id, id_a)

    end function

    function ops_cos_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_cos_simple(id_res, id_a) &
              bind(C, name='c_new_node_cos_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_cos_simple(res%id, A%id)

    end function

    function ops_rcp_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_rcp_simple(id_res, id_a) &
              bind(C, name='c_new_node_rcp_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_rcp_simple(res%id, id_a)

    end function

    function ops_rcp_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_rcp_simple(id_res, id_a) &
              bind(C, name='c_new_node_rcp_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_rcp_simple(res%id, A%id)

    end function

    function ops_sqrt_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sqrt_simple(id_res, id_a) &
              bind(C, name='c_new_node_sqrt_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_sqrt_simple(res%id, id_a)

    end function

    function ops_sqrt_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sqrt_simple(id_res, id_a) &
              bind(C, name='c_new_node_sqrt_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_sqrt_simple(res%id, A%id)

    end function

    function ops_asin_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_asin_simple(id_res, id_a) &
              bind(C, name='c_new_node_asin_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_asin_simple(res%id, id_a)

    end function

    function ops_asin_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_asin_simple(id_res, id_a) &
              bind(C, name='c_new_node_asin_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_asin_simple(res%id, A%id)

    end function

    function ops_acos_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_acos_simple(id_res, id_a) &
              bind(C, name='c_new_node_acos_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_acos_simple(res%id, id_a)

    end function

    function ops_acos_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_acos_simple(id_res, id_a) &
              bind(C, name='c_new_node_acos_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_acos_simple(res%id, A%id)

    end function

    function ops_atan_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_atan_simple(id_res, id_a) &
              bind(C, name='c_new_node_atan_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_atan_simple(res%id, id_a)

    end function

    function ops_atan_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_atan_simple(id_res, id_a) &
              bind(C, name='c_new_node_atan_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_atan_simple(res%id, A%id)

    end function

    function ops_abs_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_abs_simple(id_res, id_a) &
              bind(C, name='c_new_node_abs_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_abs_simple(res%id, id_a)

    end function

    function ops_abs_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_abs_simple(id_res, id_a) &
              bind(C, name='c_new_node_abs_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_abs_simple(res%id, A%id)

    end function

    function ops_log_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_log_simple(id_res, id_a) &
              bind(C, name='c_new_node_log_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_log_simple(res%id, id_a)

    end function

    function ops_log_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_log_simple(id_res, id_a) &
              bind(C, name='c_new_node_log_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_log_simple(res%id, A%id)

    end function

    function ops_uplus_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_uplus_simple(id_res, id_a) &
              bind(C, name='c_new_node_uplus_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_uplus_simple(res%id, id_a)

    end function

    function ops_uplus_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_uplus_simple(id_res, id_a) &
              bind(C, name='c_new_node_uplus_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_uplus_simple(res%id, A%id)

    end function

    function ops_uminus_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_uminus_simple(id_res, id_a) &
              bind(C, name='c_new_node_uminus_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_uminus_simple(res%id, id_a)

    end function

    function ops_uminus_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_uminus_simple(id_res, id_a) &
              bind(C, name='c_new_node_uminus_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_uminus_simple(res%id, A%id)

    end function

    function ops_log10_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_log10_simple(id_res, id_a) &
              bind(C, name='c_new_node_log10_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_log10_simple(res%id, id_a)

    end function

    function ops_log10_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_log10_simple(id_res, id_a) &
              bind(C, name='c_new_node_log10_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_log10_simple(res%id, A%id)

    end function

    function ops_tanh_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_tanh_simple(id_res, id_a) &
              bind(C, name='c_new_node_tanh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_tanh_simple(res%id, id_a)

    end function

    function ops_tanh_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_tanh_simple(id_res, id_a) &
              bind(C, name='c_new_node_tanh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_tanh_simple(res%id, A%id)

    end function

    function ops_sinh_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sinh_simple(id_res, id_a) &
              bind(C, name='c_new_node_sinh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_sinh_simple(res%id, id_a)

    end function

    function ops_sinh_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_sinh_simple(id_res, id_a) &
              bind(C, name='c_new_node_sinh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_sinh_simple(res%id, A%id)

    end function

    function ops_cosh_array(A) result(res)
      implicit none

      interface
         subroutine c_new_node_cosh_simple(id_res, id_a) &
              bind(C, name='c_new_node_cosh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(array), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_array_simple(A%ptr, id_a)
      call c_new_node_cosh_simple(res%id, id_a)

    end function

    function ops_cosh_node(A) result(res)
      implicit none

      interface
         subroutine c_new_node_cosh_simple(id_res, id_a) &
              bind(C, name='c_new_node_cosh_simple')
           use iso_c_binding
           integer:: id_res, id_a
         end subroutine
      end interface

      type(node), intent(in) :: A
      type(node) :: res
      integer:: id_a
      call c_new_node_cosh_simple(res%id, A%id)

    end function


    function ops_pow_node_int (A, B) result(C)
      implicit none
      type(node), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      id_a = A%id

      call c_new_seqs_scalar_node_int_simple(B, id_b)
      call c_new_node_pow_simple(C%id,A%id, id_b )
    end function
    function ops_pow_node_float (A, B) result(C)
      implicit none
      type(node), intent(in) :: A
      real, intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      id_a = A%id

      call c_new_seqs_scalar_node_float_simple(B, id_b)
      call c_new_node_pow_simple(C%id,A%id, id_b )
    end function
    function ops_pow_node_double (A, B) result(C)
      implicit none
      type(node), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      id_a = A%id

      call c_new_seqs_scalar_node_double_simple(B, id_b)
      call c_new_node_pow_simple(C%id,A%id, id_b )
    end function
    function ops_pow_array_int (A, B) result(C)
      implicit none
      type(array), intent(in) :: A
      integer, intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_int_simple(B, id_b)
      call c_new_node_pow_simple(C%id, id_a, id_b)
    end function
    function ops_pow_array_float (A, B) result(C)
      implicit none
      type(array), intent(in) :: A
      real, intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_float_simple(B, id_b)
      call c_new_node_pow_simple(C%id, id_a, id_b)
    end function
    function ops_pow_array_double (A, B) result(C)
      implicit none
      type(array), intent(in) :: A
      real(8), intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow_simple(id_res, id_a, id_b) &
              bind(C, name='c_new_node_pow_simple')
           use iso_c_binding
          integer :: id_res, id_a, id_b
         end subroutine
      end interface
      integer:: id_a, id_b
      call c_new_node_array_simple(A%ptr, id_a)

      call c_new_seqs_scalar_node_double_simple(B, id_b)
      call c_new_node_pow_simple(C%id, id_a, id_b)
    end function

       
    subroutine array_assign_array(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(array), intent(in) :: B

      call c_array_assign_array(A%ptr, B%ptr, A%lr, B%lr)

      call try_destroy(B)
    end subroutine

    subroutine node_assign_node(A, B)
      implicit none
      type(node), intent(inout) :: A
      type(node), intent(in) :: B

      call c_node_assign_node(A%ptr, B%ptr)

      call try_destroy(B)
    end subroutine

    subroutine node_assign_array(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(node), intent(in) :: B

      call c_node_assign_array(A%ptr)

    end subroutine

    subroutine eval(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(node), intent(in) :: B

      call c_node_assign_array(A%ptr)

    end subroutine

    subroutine set_stencil(st, sw)
      implicit none
      integer, value :: st, sw
      
      default_stencil_width = sw;
      default_stencil_type = st;
    end subroutine

    subroutine set_data_type(dt)
      implicit none
      integer, value :: dt
      
      default_data_type = dt
    end subroutine

    subroutine set_rvalue_array(A)
      implicit none
      type(array), intent(inout) :: A

      A%lr = RVALUE
    end subroutine

    subroutine set_rvalue_node(A)
      implicit none
      type(node), intent(inout) :: A

      A%lr = RVALUE
    end subroutine
    
    subroutine set_lvalue(A)
      implicit none
      type(array), intent(inout) :: A

      A%lr = LVALUE
    end subroutine
    
    function make_psudo3d(A) result(B)
      implicit none
      type(array),intent(in) :: A
      type(array) :: B
      interface
         subroutine c_make_psudo3d(dst, src) &
              bind(C, name='c_make_psudo3d')
           use iso_c_binding
           implicit none
           type(c_ptr), intent(inout) :: dst           
           type(c_ptr), intent(in) :: src
         end subroutine
      end interface

      call c_make_psudo3d(B%ptr, A%ptr)

      call set_rvalue(B)

      call try_destroy(A)
    end function

    function has_nan_or_inf(A, d) result(res)
      implicit none
      real :: name
      type(array), intent(in) :: A
      integer, optional :: d
      logical :: res
      integer :: i
      interface
         subroutine c_has_nan_or_inf(i, A, d) &
              bind(C, name = 'c_has_nan_or_inf')
           use iso_c_binding
           implicit none
           type(c_ptr), intent(in) :: A
           integer, value :: d
           integer :: i
         end subroutine
      end interface

      if(present(d)) then
         call c_has_nan_or_inf(i, A%ptr, d)
      else
         call c_has_nan_or_inf(i, A%ptr, 1)
      end if

      if(i == 0) then
         res = .false.
      else
         res = .true.
      endif
      
    end function

  end module
