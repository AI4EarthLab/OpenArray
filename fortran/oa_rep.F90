#include "config.h"
  
  ///:mute
  ///:include "../NodeTypeF.fypp"
  ///:set types = [['double','real(8)', 'scalar'],&
       ['float','real', 'scalar'], &
       ['int','integer', 'scalar'], &
       ['array', 'type(array)', 'array'], &
       ['node', 'type(node)',  'node']]
  ///:endmute

module oa_rep
  use iso_c_binding
  use oa_type
  interface rep
     ///:for t in ['node', 'array']
     module procedure rep_${t}$
     ///:endfor
  end interface rep  
contains

  ///:for t in ['node', 'array']
  function rep_${t}$(A, x, y, z) result(B)
    implicit none
    type(${t}$) :: A
    type(node) :: B
    integer :: x
    integer, optional :: y, z
    type(node) :: ND, NA
    integer :: x1,y1,z1

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
    
    ND = new_local_int3([x1,y1,z1])

    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_op2(B%ptr, TYPE_REP, NA%ptr, ND%ptr)
    ///:else
    call c_new_node_op2(B%ptr, TYPE_REP, A%ptr, ND%ptr)    
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    call destroy(ND)
  end function
  ///:endfor
end module
