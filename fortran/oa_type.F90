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
    
    interface consts
       ///:for t in TYPE
       module procedure consts_${t[0]}$
       ///:endfor
    end interface consts

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
       subroutine c_new_node_op1(A, nodetype, U) &
            bind(C, name='c_new_node_op1')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: U 
         integer(c_int), intent(in), VALUE :: nodetype
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
 
    integer, parameter :: OA_INT    = 0
    integer, parameter :: OA_FLOAT  = 1
    integer, parameter :: OA_DOUBLE = 2

    integer, parameter :: STENCIL_STAR = 0
    integer, parameter :: STENCIL_BOX  = 1
    
    integer :: default_data_type     = OA_FLOAT
    integer :: default_stencil_type  = STENCIL_BOX    
    integer :: default_stencil_width = 1

  contains

    subroutine destroy_array(A)
      use iso_c_binding
      type(Array), intent(inout) :: A

      interface
         subroutine c_destroy_array(A) &
              bind(C, name = 'c_destroy_array')
           use iso_c_binding
           type(c_ptr), intent(in) :: A
         end subroutine
      end interface

      call c_destroy_array(A%ptr)

    end subroutine

    subroutine destroy_node(A)
      use iso_c_binding 
      type(Node), intent(inout) :: A

      interface
         subroutine c_destroy_node(A) &
              bind(C, name = 'c_destroy_node')
           use iso_c_binding
           type(c_ptr), intent(in) :: A
         end subroutine
      end interface
      
      call c_destroy_node(A%ptr)

      !A%ptr = C_NULL_PTR
    end subroutine

    function string_f2c(f_string) result(c_string)
      use iso_c_binding
      character(len=*):: f_string
      CHARACTER(LEN=LEN_TRIM(f_string)+1,KIND=C_CHAR) :: c_string

      c_string = trim(f_string) // C_NULL_CHAR
    end function
    
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

      call c_display_node(A%ptr, prefix)
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
      !call c_new_node_op1(B%ptr, ${n[0]}$,NA%ptr)
      
      call c_new_node_${name}$(B%ptr, NA%ptr)
      ///:else
      call c_new_node_${name}$(B%ptr, A%ptr)
      ///:endif
      !!B%ptr = C_NULL_PTR
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
      
      call c_shape_array(A%ptr, res)   
    end function

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
    end function
    ///:endfor
    ///:endfor

       
    subroutine array_assign_array(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(array), intent(in) :: B

      call c_array_assign_array(A%ptr, B%ptr, A%lr, B%lr)
      
    end subroutine

    subroutine node_assign_node(A, B)
      implicit none
      type(node), intent(inout) :: A
      type(node), intent(in) :: B

      call c_node_assign_node(A%ptr, B%ptr)

    end subroutine

    subroutine node_assign_array(A, B)
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

    subroutine set_rvalue(A)
      implicit none
      type(array), intent(inout) :: A

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
    end function
  end module
