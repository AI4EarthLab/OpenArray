
#include "config.h"

module oa_shift
  use iso_c_binding
  use oa_type
  
  interface shift
     module procedure shift_node
     module procedure shift_array
  end interface shift

  interface circshift
     module procedure circshift_node     
     module procedure circshift_array     
  end interface 
  
contains

  

  function shift_node(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_shift_simple(id_b, id_a, id_int3) &
            bind(C, name = 'c_new_node_shift_simple')
         use iso_c_binding
        implicit none
        integer:: id_a, id_b, id_int3
       end subroutine
    end interface
    
    type(node) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    integer:: id_int3

    if(present(y)) then
       op_y = y
    else
       op_y = 0
    end if

    if(present(z)) then
       op_z = z
    else
       op_z = 0
    end if

    call c_new_node_int3_simple_shift(x, op_y, op_z, id_int3)
 
    call c_new_node_shift_simple(B%id, A%id, id_int3)

  end function

  function circshift_node(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_circshift_simple(id_b, id_a, id_int3) &
            bind(C, name = 'c_new_node_circshift_simple')
         use iso_c_binding
        implicit none
        integer:: id_a, id_b, id_int3
       end subroutine
    end interface
    
    type(node) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    integer:: id_int3

    if(present(y)) then
       op_y = y
    else
       op_y = 0
    end if

    if(present(z)) then
       op_z = z
    else
       op_z = 0
    end if

    call c_new_node_int3_simple_shift(x, op_y, op_z, id_int3)
 
    call c_new_node_circshift_simple(B%id, A%id, id_int3)

  end function


  function shift_array(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_shift_simple(id_b, id_a, id_int3) &
            bind(C, name = 'c_new_node_shift_simple')
         use iso_c_binding
            implicit none
            integer:: id_a, id_b, id_int3
       end subroutine
    end interface

    type(array) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    integer :: id1, id2

    if(present(y)) then
       op_y = y
    else
       op_y = 0
    end if

    if(present(z)) then
       op_z = z
    else
       op_z = 0
    end if

    call c_new_node_array_simple(A%ptr, id1)

    call c_new_node_int3_simple_shift(x, op_y, op_z, id2)
 
    call c_new_node_shift_simple(B%id, id1,id2)

  end function

  function circshift_array(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_circshift_simple(id_b, id_a, id_int3) &
            bind(C, name = 'c_new_node_circshift_simple')
         use iso_c_binding
            implicit none
            integer:: id_a, id_b, id_int3
       end subroutine
    end interface

    type(array) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    integer :: id1, id2

    if(present(y)) then
       op_y = y
    else
       op_y = 0
    end if

    if(present(z)) then
       op_z = z
    else
       op_z = 0
    end if

    call c_new_node_array_simple(A%ptr, id1)

    call c_new_node_int3_simple_shift(x, op_y, op_z, id2)
 
    call c_new_node_circshift_simple(B%id, id1,id2)

  end function
  
end module
