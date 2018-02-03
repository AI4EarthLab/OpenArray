#include "config.h"
  
  ///:mute
  ///:include "../NodeTypeF.fypp"
  ///:set types = [['double','real(8)', 'scalar'],&
       ['float','real', 'scalar'], &
       ['int','integer', 'scalar'], &
       ['array', 'type(array)', 'array'], &
       ['node', 'type(node)',  'node']]
  ///:endmute

module oa_sum
  use iso_c_binding
  use oa_type

  interface csum
     ///:for t in ['node', 'array']
     module procedure csum_${t}$
     ///:endfor
  end interface csum

  interface sum
     ///:for t in ['node', 'array']
     module procedure sum_${t}$
     ///:endfor
  end interface sum  
contains

  ///:for t in ['node', 'array']
  function csum_${t}$(A, d) result(B)
    implicit none
    type(${t}$) :: A
    type(node) :: B
    integer :: d
    type(node) :: ND, NA

    call c_new_seqs_scalar_node_int(ND%ptr, d)
    
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    ! call c_new_node_op2(B%ptr, TYPE_CSUM, NA%ptr, ND%ptr)
    call c_new_node_csum(B%ptr, NA%ptr, ND%ptr)
    ///:else
    ! call c_new_node_op2(B%ptr, TYPE_CSUM, A%ptr, ND%ptr)    
    call c_new_node_csum(B%ptr, A%ptr, ND%ptr)    
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    call destroy(ND)    
    
  end function
  ///:endfor


  ///:for t in ['node', 'array']
  function sum_${t}$(A, d) result(B)
    implicit none
    type(${t}$) :: A
    type(node) :: B
    integer, optional :: d
    type(node) :: ND, NA
    integer :: d1

    d1 = -1
    if(present(d)) then
       d1 = d
    end if
    
    call c_new_seqs_scalar_node_int(ND%ptr, d1)
    
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    ! call c_new_node_op2(B%ptr, TYPE_SUM, NA%ptr, ND%ptr)
    call c_new_node_sum(B%ptr, NA%ptr, ND%ptr)
    ///:else
    ! call c_new_node_op2(B%ptr, TYPE_SUM, A%ptr, ND%ptr)    
    call c_new_node_sum(B%ptr, A%ptr, ND%ptr)    
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    call destroy(ND)    
    
  end function
  ///:endfor
end module
