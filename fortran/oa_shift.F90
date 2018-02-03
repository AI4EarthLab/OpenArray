
#include "config.h"

module oa_shift
  use iso_c_binding
  use oa_type
  
  interface shift
     ///:for t in ['node', 'array']
     module procedure shift_${t}$
     ///:endfor
  end interface shift

  interface circshift
     ///:for t in ['node', 'array']
     module procedure circshift_${t}$     
     ///:endfor
  end interface 
  
contains

  ///:for t in ['node', 'array']
  function shift_${t}$(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_shift(o, u, v) &
            bind(C, name = 'c_new_node_shift')
         use iso_c_binding
         type(c_ptr), intent(inout) :: o
         type(c_ptr), intent(in) :: u, v
       end subroutine
    end interface
    
    type(${t}$) :: A
    type(node)  :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    type(node) :: ND, NA

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

    ND = new_local_int3([x, op_y, op_z])
    
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_shift(B%ptr, NA%ptr, ND%ptr)
    ///:else
    call c_new_node_shift(B%ptr, A%ptr,  ND%ptr)
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    call destroy(ND)
  end function
  ///:endfor


  ///:for t in ['node', 'array']
  function circshift_${t}$(A, x, y, z) result(B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_new_node_circshift(o, u, v) &
            bind(C, name = 'c_new_node_circshift')
         use iso_c_binding
         type(c_ptr), intent(inout) :: o
         type(c_ptr), intent(in) :: u, v
       end subroutine
    end interface
    
    type(${t}$) :: A
    type(node)  :: B
    integer :: x
    integer, optional :: y, z
    integer :: op_y, op_z
    type(node) :: ND, NA

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

    ND = new_local_int3([x, op_y, op_z])
    
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_circshift(B%ptr, NA%ptr, ND%ptr)
    ///:else
    call c_new_node_circshift(B%ptr, A%ptr,  ND%ptr)
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    call destroy(ND)
  end function
  ///:endfor
  
end module
