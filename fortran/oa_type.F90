#include "config.h"
  ///:include "../NodeTypeF.fypp"
  ///:mute
  ///:set TYPE = [['int', 'integer'], &
       ['float',  'real'], ['double', 'real(kind=8)']]
  ///:endmute

#include "config.h"
  
  module oa_type
    use iso_c_binding
    use mpi
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
       ///:for t in TYPE
       module procedure consts_${t[0]}$
       ///:endfor
    end interface consts

    interface local_sub
      ///:for t in TYPE
      module procedure local_sub_${t[0]}$
      ///:endfor
    end interface local_sub

    interface set_local
      ///:for t in TYPE
      module procedure set_local_${t[0]}$
      ///:endfor
    end interface set_local

    ///:for t in TYPE
    interface
       subroutine c_new_seqs_scalar_node_${t[0]}$ &
            (ptr, val) &
            bind(C, name = 'c_new_seqs_scalar_node_${t[0]}$')
         use iso_c_binding
         type(c_ptr), intent(inout) :: ptr
         ${t[1]}$, intent(in), VALUE :: val
       end subroutine
    end interface
    ///:endfor

    interface
       subroutine c_new_node_array(A, B) &
            bind(C, name='c_new_node_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
       end subroutine
    end interface

    interface
       subroutine c_new_node_op2(A, nodetype, U, V) &
            bind(C, name='c_new_node_op2')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: U, V 
         integer(c_int), intent(in), VALUE :: nodetype
       end subroutine
    end interface

    interface
       subroutine c_new_node_csum(A, U, V) &
            bind(C, name='c_new_node_csum')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: U, V 
       end subroutine
    end interface
    
    interface
       subroutine c_new_node_sum(A, U, V) &
            bind(C, name='c_new_node_sum')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: U, V 
       end subroutine
    end interface

    ///:for n in [i for i in L if i[3] == 'D']
    interface ${n[2]}$    
    ///:for t in ['node', 'array']
       module procedure ${n[2]}$_${t}$
    ///:endfor
    end interface ${n[2]}$    
    ///:endfor

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
       subroutine c_node_assign_array(A, B) &
            bind(C, name='c_node_assign_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
       end subroutine
    end interface

    ///:for op in [o for o in L if o[3] in ['A','B','F']]
    interface operator (${op[2]}$)
       ///:for type1 in types
       ///:for type2 in types
       ///:if not (type1[2] == 'scalar' and type2[2] == 'scalar')
       module procedure ops_${type1[0]}$_${op[1]}$_${type2[0]}$
       ///:endif
       ///:endfor
       ///:endfor
    end interface operator (${op[2]}$)
    ///:endfor

    ///:for op in [o for o in L if o[3] == 'C']
    ///:set b = 'operator ({0})'.format(op[2]) &
         if (op[2] in ['+', '-']) else op[2]
    interface ${b}$ 
       ///:for type1 in types
       ///:if not (type1[2] == 'scalar')
       module procedure ops_${op[1]}$_${type1[2]}$
       ///:endif
       ///:endfor
    end interface ${b}$ 
    ///:endfor

    interface operator(**)
       ///:for t1 in ['node', 'array']
       ///:for t2 in scalar_dtype
       module procedure ops_pow_${t1}$_${t2[0]}$
       ///:endfor
       ///:endfor
    end interface operator(**)
    
    interface get_local_buffer
       ///:for t in scalar_dtype
       module procedure get_local_buffer_${t[0]}$
       ///:endfor
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

    ///:mute
    ///:set NAME = [['ones'], ['zeros'], ['rands'], ['seqs']]
    ///:endmute
    ///:for t in NAME
    function ${t[0]}$(m, n, k, sw, dt) result(A)
      integer(c_int) :: m, op_n, op_k, op_sw, op_dt
      integer(c_int), optional :: n, k, sw, dt
      type(Array) :: A

      interface
         subroutine c_${t[0]}$(ptr, m, n, k, &
              op_sw, op_dt) &
              bind(C, name = 'c_${t[0]}$')
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
      
      call c_${t[0]}$(A%ptr, m, op_n, op_k, op_sw, op_dt)
      
      call set_rvalue(A)
      
    end function
    ///:endfor

    ///:for t in TYPE
    function consts_${t[0]}$(m, n, k, val, sw) result(A)
      integer(c_int) :: m, n, k, op_sw
      integer(c_int), optional :: sw
      ${t[1]}$ :: val
      type(Array) :: A

      interface
         subroutine c_consts_${t[0]}$ &
              (ptr, m, n, k, val, op_sw) &
              bind(C, name='c_consts_${t[0]}$')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw
           ${t[1]}$, intent(in), VALUE :: val
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = default_stencil_width
      endif

      call c_consts_${t[0]}$(A%ptr, &
           m, n, k, val, op_sw)
      call set_rvalue(A)
    end function
    ///:endfor

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


    ///:for t in ['node', 'array']
    ///:for n in [i for i in L if i[3] == 'D']
    ///:set name = n[1]
    function ${n[2]}$_${t}$(A) result(B)
      implicit none

      interface
         subroutine c_new_node_${name}$(A, U) &
              bind(C, name='c_new_node_${name}$')
           use iso_c_binding
           type(c_ptr), intent(inout) :: A
           type(c_ptr), intent(in) :: U 
         end subroutine
      end interface

      type(${t}$) :: A
      type(node)  :: B
      type(node) :: NA

      ///:if t == 'array'
      call c_new_node_array(NA%ptr, A%ptr)
      
      call c_new_node_${name}$(B%ptr, NA%ptr)
      ///:else
      call c_new_node_${name}$(B%ptr, A%ptr)
      ///:endif
      !!B%ptr = C_NULL_PTR

      call set_rvalue(B)
      
      call try_destroy(A)
      call destroy(NA)
    end function
    ///:endfor
    ///:endfor

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

    ///:for t in scalar_dtype
    subroutine get_local_buffer_${t[0]}$(res, A)
      use iso_c_binding
      implicit none
      type(array) :: A
      ${t[1]}$, dimension(:,:,:), pointer, intent(out) :: res
      type(c_ptr) :: tmp
      integer :: s(3)
      integer :: flag
      interface
         subroutine c_is_array_${t[0]}$(flag, A) &
              bind(C, name = "c_is_array_${t[0]}$")
           use iso_c_binding
           type(c_ptr) :: A
           integer(c_int) :: flag
         end subroutine
      end interface

      !ASSERT_LVALUE(A)
      
      tmp = get_buffer_ptr(A)
      s   = buffer_shape(A)

      call c_is_array_${t[0]}$ (flag, A%ptr)
      if(flag > 0) then
         call c_f_pointer(tmp, res, [s(1), s(2), s(3)])
      else
         print*, "Error: pointer does not match array's data type"
         stop
      end if
    end subroutine
    ///:endfor
    
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

    ///:for t in TYPE
    subroutine local_sub_${t[0]}$(A, x, y, z, val)
      implicit none
      interface
        subroutine c_local_sub_${t[0]}$(a, x, y, z, s) &
          bind(C, name = "c_local_sub_${t[0]}$")
        use iso_c_binding
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           ${t[1]}$, intent(out) :: s
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      ${t[1]}$ :: val
      rx = x - 1
      ry = y - 1
      rz = z - 1

      call c_local_sub_${t[0]}$(A%ptr, rx, ry, rz, val)

    end subroutine

    ///:endfor

    ///:for t in TYPE
    subroutine set_local_${t[0]}$(A, x, y, z, val)
      implicit none
      interface
        subroutine c_set_local_${t[0]}$(a, x, y, z, val) &
          bind(C, name = "c_set_local_${t[0]}$") 
        use iso_c_binding 
           implicit none
           type(c_ptr) :: a
           integer :: x, y, z
           ${t[1]}$ :: val
         end subroutine
      end interface

      type(array) :: A
      integer :: x, y, z
      integer :: rx, ry, rz
      ${t[1]}$ :: val
      rx = x - 1 
      ry = y - 1
      rz = z - 1

      call c_set_local_${t[0]}$(A%ptr, rx, ry, rz, val)

    end subroutine 
    
    ///:endfor

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


    ///:for e in [x for x in L if x[3] in ['A','B','F']]
    ///:set op = e[1]
    ///:for type1 in types
    ///:for type2 in types
    ///:if not (type1[2] == 'scalar' and type2[2] == 'scalar')
    function ops_${type1[0]}$_${op}$_${type2[0]}$(A, B) &
         result(res)
      implicit none

      interface
         subroutine c_new_node_${op}$(p1, p2, p3) &
              bind(C, name='c_new_node_${op}$')
           use iso_c_binding
           implicit none
           type(c_ptr), intent(inout) :: p1
           type(c_ptr), intent(in) :: p2, p3
         end subroutine
      end interface

      ${type1[1]}$, intent(in) :: A
      ${type2[1]}$, intent(in) :: B
      ///:if type1[0] != 'node'
      type(node) :: C
      ///:endif
      ///:if type2[0] != 'node'
      type(node) :: D
      ///:endif
      type(node) :: res

      ///:if type1[0] == 'node'
      ///:set AC = 'A'
      ///:else
      ///:if type1[2] == 'scalar'
      call c_new_seqs_scalar_node_${type1[0]}$(C%ptr, A)
      ///:else
      call c_new_node_array(C%ptr, A%ptr)
      ///:endif
      ///:set AC = 'C'
      ///:endif

      ///:if type2[0] == 'node'
      ///:set BD = 'B'
      ///:else
      ///:if type2[2] == 'scalar'
      call c_new_seqs_scalar_node_${type2[0]}$(D%ptr, B)
      ///:else
      call c_new_node_array(D%ptr, B%ptr)
      ///:endif
      ///:set BD = 'D'
      ///:endif

      call c_new_node_${op}$(res%ptr, ${AC}$%ptr, ${BD}$%ptr)

      call set_rvalue(res)

      ///:if type1[2] != 'scalar'
      call try_destroy(A)
      ///:endif

      ///:if type2[2] != 'scalar'
      call try_destroy(B)
      ///:endif
      
      ///:if type1[0] != 'node'
      call destroy(C)
      ///:endif
      
      ///:if type2[0] != 'node'
      call destroy(D)
      ///:endif
      
    end function

    ///:endif
    ///:endfor
    ///:endfor
    ///:endfor


    ///:for e in [x for x in L if x[3] == 'C']
    ///:set op = e[1]
    ///:for type1 in types
    ///:if not (type1[2] == 'scalar')
    function ops_${op}$_${type1[2]}$(A) result(res)
      implicit none

      interface
         subroutine c_new_node_${op}$(A, U) &
              bind(C, name='c_new_node_${op}$')
           use iso_c_binding
           type(c_ptr), intent(inout) :: A
           type(c_ptr), intent(in) :: U 
         end subroutine
      end interface

      ${type1[1]}$, intent(in) :: A
      type(node) :: res
      ///:if type1[0] != 'node'
      type(node) :: C
      ///:endif
      ///:if type1[0] == 'node'
      ///:set AC = 'A'
      ///:else
      call c_new_node_array(C%ptr, A%ptr)
      ///:set AC = 'C'
      ///:endif
      call c_new_node_${op}$(res%ptr, ${AC}$%ptr)

      call set_rvalue(res)

      call try_destroy(A)

      ///:if type1[0] != 'node'
      call destroy(C)
      ///:endif

    end function

    ///:endif
    ///:endfor
    ///:endfor

    ///:for t1 in ['node', 'array']
    ///:for t2 in scalar_dtype
    function ops_pow_${t1}$_${t2[0]}$ (A, B) result(C)
      implicit none
      type(${t1}$), intent(in) :: A
      ${t2[1]}$, intent(in) :: B
      type(node) :: C, NA, NB

      interface
         subroutine c_new_node_pow(A, U, V) &
              bind(C, name='c_new_node_pow')
           use iso_c_binding
           type(c_ptr), intent(inout) :: A
           type(c_ptr), intent(in) :: U
           type(c_ptr), intent(in) :: V           
         end subroutine
      end interface

      ///:if t1 == 'node'
      ///:set A = 'A'
      ///:else
      call c_new_node_array(NA%ptr, A%ptr)
      ///:set A = 'NA'
      ///:endif

      call c_new_seqs_scalar_node_${t2[0]}$(NB%ptr, B)

      call c_new_node_pow(C%ptr, ${A}$%ptr, NB%ptr)

      call set_rvalue(C)

      call try_destroy(A)
      call destroy(NA)
      call destroy(NB)
    end function
    ///:endfor
    ///:endfor

       
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

      call c_node_assign_array(A%ptr, B%ptr)

      call try_destroy(B)
    end subroutine

    subroutine eval(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(node), intent(in) :: B

      call c_node_assign_array(A%ptr, B%ptr)

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
