#include "config.h"
  ///:include "NodeTypeF.fypp"
  ///:mute
  ///:set TYPE = [['int', 'integer'], &
       ['float',  'real'], ['double', 'real(kind=8)']]
  ///:endmute

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

    interface display
       module procedure display_node
       module procedure display_array
    end interface display

    interface consts
       ///:for t in TYPE
       module procedure consts_${t[0]}$
       ///:endfor
    end interface consts

    ///:for t in TYPE
    interface
       subroutine c_new_seqs_scalar_node_${t[0]}$(ptr, val, comm) &
            bind(C, name = 'c_new_seqs_scalar_node_${t[0]}$')
         use iso_c_binding
         type(c_ptr), intent(inout) :: ptr
         ${t[1]}$, intent(in), VALUE :: val
         integer(c_int), intent(in), VALUE :: comm
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
    
    integer, parameter :: OA_INT = 0
    integer, parameter :: OA_FLOAT = 1
    integer, parameter :: OA_DOUBLE = 2
    
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
    function ${t[0]}$(m, n, k, sw, dt, comm) result(A)
      integer(c_int) :: m, n, k, op_sw, op_dt, op_comm
      integer(c_int), optional :: sw, dt, comm
      type(Array) :: A

      interface
         subroutine c_${t[0]}$(ptr, m, n, k, op_sw, op_dt, op_comm) &
              bind(C, name = 'c_${t[0]}$')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, &
                op_sw, op_dt, op_comm
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = STENCIL_WIDTH
      endif

      if (present(dt)) then
         op_dt = dt
      else
         op_dt = DATA_TYPE
      endif

      if (present(comm)) then
         op_comm = comm
      else
         op_comm = MPI_COMM_WORLD
      endif

      call c_${t[0]}$(A%ptr, m, n, k, op_sw, op_dt, op_comm)
      A%lr = R

    end function
    ///:endfor

    ///:for t in TYPE
    function consts_${t[0]}$(m, n, k, val, sw, comm) result(A)
      integer(c_int) :: m, n, k, op_sw, op_comm
      integer(c_int), optional :: sw, comm
      ${t[1]}$ :: val
      type(Array) :: A

      interface
         subroutine c_consts_${t[0]}$(ptr, m, n, k, val, op_sw, op_comm) &
              bind(C, name='c_consts_${t[0]}$')
           use iso_c_binding
           type(c_ptr), intent(inout) :: ptr
           integer(c_int), intent(in), VALUE :: m, n, k, op_sw, op_comm
           ${t[1]}$, intent(in), VALUE :: val
         end subroutine
      end interface


      if (present(sw)) then
         op_sw = sw
      else
         op_sw = STENCIL_WIDTH
      endif

      if (present(comm)) then
         op_comm = comm
      else
         op_comm = MPI_COMM_WORLD
      endif

      call c_consts_${t[0]}$(A%ptr, m, n, k, val, op_sw, op_comm)
      A%lr = R
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
  end module
