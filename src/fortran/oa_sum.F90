#include "config.h"
  

module oa_sum
  use iso_c_binding
  use oa_type

  interface csum
     module procedure csum_node
     module procedure csum_array
  end interface csum

  interface sum
     module procedure sum_node
     module procedure sum_array
  end interface sum  
contains

  function csum_node(A, d) result(B)
    implicit none
    type(node) :: A
    type(node) :: B
    integer :: d, id_d

    call c_new_seqs_scalar_node_int_simple(d, id_d) 
    call c_new_node_csum_simple(B%id, A%id, id_d)    
    
  end function

  function csum_array(A, d) result(B)
    implicit none
    type(array) :: A
    type(node) :: B
    integer :: d, id_a, id_d

    call c_new_node_array_simple(A%ptr,id_a )
    call c_new_seqs_scalar_node_int_simple(d, id_d)

    call c_new_node_csum_simple(B%id, id_a, id_d)

  end function

  function sum_node(A, d) result(B)
    implicit none
    type(node) :: A
    type(node) :: B
    integer, optional :: d
    integer :: d1, id_d

    d1 = -1
    if(present(d)) then
       d1 = d
    end if
    
    call c_new_seqs_scalar_node_int_simple(d1, id_d)
    call c_new_node_sum_simple(B%id, A%id, id_d)    

  end function

  function sum_array(A, d) result(B)
    implicit none
    type(array) :: A
    type(node) :: B
    integer, optional :: d
    integer :: d1, id_a, id_d1

    d1 = -1
    if(present(d)) then
       d1 = d
    end if
    
    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_seqs_scalar_node_int_simple(d1, id_d1)

    call c_new_node_sum_simple(B%id, id_a, id_d1)

  end function
end module
