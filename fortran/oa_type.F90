#include "config.h"

module oa_type
  use iso_c_binding
  use mpi
  type Array
     type(c_ptr) :: ptr = C_NULL_PTR
     integer :: pos
   contains
     final :: destroy_array
  end type Array

  type Node
     type(c_ptr) :: ptr = C_NULL_PTR
     integer :: pos
   contains
     final :: destroy_node
  end type Node

  interface
    subroutine c_destroy_array(A) bind(C, name = 'destroy_array')
      use iso_c_binding
      type(c_ptr), intent(in), VALUE :: A
    end subroutine
  end interface

  interface
    subroutine c_destroy_node(A) bind(C, name = 'destroy_node')
      use iso_c_binding
      type(c_ptr), intent(in), VALUE :: A
    end subroutine
  end interface

  interface
    subroutine display_array(A) bind(C, name = 'display_array')
      use iso_c_binding
      type(c_ptr), intent(in), VALUE :: A
    end subroutine
  end interface
  
  interface assignment(=)
    module procedure array_assign_array
  end interface assignment(=)


  ///:mute
  ///:set NAME = [['ones'], ['zeros'], ['rands'], ['seqs']]
  ///:endmute
  ///:for t in NAME
  interface
    subroutine c_${t[0]}$(ptr, m, n, k, st, dt, comm) &
        bind(C, name = '${t[0]}$')
      use iso_c_binding
      type(c_ptr), intent(inout) :: ptr
      integer(c_int), intent(in), VALUE :: m, n, k, st, dt, comm
    end subroutine
  end interface

  ///:endfor

  interface
    subroutine c_array_assign_array(A, B, pa, pb) &
        bind(C, name='array_assign_array')
      use iso_c_binding
      type(c_ptr), intent(inout) :: A
      type(c_ptr), intent(in) :: B
      integer(c_int), intent(inout) :: pa
      integer(c_int), intent(in) :: pb
    end subroutine
  end interface

///:mute
///:set TYPE = [['int', 'integer'], &
       ['float',  'real'], &
       ['double', 'real(kind=8)']]
///:endmute
///:for t in TYPE
///:endfor


contains

  subroutine array_assign_array(A, B)
    implicit none
    type(Array), intent(inout) :: A
    type(Array), intent(in) :: B

    call c_array_assign_array(A%ptr, B%ptr, A%pos, B%pos)
    !print *, A%ptr, B%ptr, A%pos, B%pos
  end subroutine

  subroutine destroy_array(A)
    use iso_c_binding
    type(Array), intent(inout) :: A
    !if (A%ptr /= C_NULL_PTR) then
      call c_destroy_array(A%ptr)
      A%ptr = C_NULL_PTR
    !endif
  end subroutine

  subroutine destroy_node(A)
    type(Node), intent(inout) :: A

    !call c function to destroy node here.
    A%ptr = C_NULL_PTR
  end subroutine
  
  ///:for t in NAME
  function ${t[0]}$(m, n, k, op_st, op_dt, op_comm) result(A)
    integer(c_int) :: m, n, k, st, dt, comm
    integer(c_int), optional :: op_st, op_dt, op_comm
    type(Array) :: A

    if (present(op_st)) then
      st = op_st
    else
      st = STENCIL_WIDTH
    endif

    if (present(op_dt)) then
      dt = op_dt
    else
      dt = DATA_TYPE
    endif

    if (present(op_comm)) then
      comm = op_comm
    else
      comm = MPI_COMM_WORLD
    endif

    call c_${t[0]}$(A%ptr, m, n, k, st, dt, comm)
    A%pos = R

  end function
  ///:endfor


///:for t in [['int', 'integer'], &
       ['real',  'real'], &
       ['real8', 'real(kind=8)'], &
       ['array', 'type(array)']]
  subroutine create_node_${t[0]}$(B, A)
    type(node), intent(out) :: B
    type(array) :: A

    !call c function to create a node
  end subroutine

///:endfor
  
end module
