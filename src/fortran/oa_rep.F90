#include "config.h"
  

module oa_rep
  use iso_c_binding
  use oa_type
  interface rep
     module procedure rep_node
     module procedure rep_array
  end interface rep  
contains

  function rep_node(A, x, y, z) result(B)
    implicit none
    type(node) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    integer :: x1,y1,z1
    integer :: id_int3
    x1 = x

    if(present(y)) then
       y1 = y
    else
       y1 = 1
    end if

    if(present(z)) then
       z1 = z
    else
       z1 = 1
    end if

    call c_new_node_int3_simple_rep(x1, y1, z1, id_int3)
    call c_new_node_rep_simple(B%id, A%id, id_int3)    

  end function

  function rep_array(A, x, y, z) result(B)
    implicit none
    type(node) :: B
    type(array) :: A
    integer :: x
    integer, optional :: y, z
    integer :: x1,y1,z1
    integer :: id_a, id_int3 
    x1 = x

    if(present(y)) then
       y1 = y
    else
       y1 = 1
    end if

    if(present(z)) then
       z1 = z
    else
       z1 = 1
    end if
    
    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_int3_simple_rep(x1, y1, z1, id_int3)

    call c_new_node_rep_simple(B%id, id_a, id_int3)

  end function
end module
