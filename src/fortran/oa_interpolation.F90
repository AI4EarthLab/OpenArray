#include "config.h"
 !
module oa_interpolation
 ! use iso_c_binding
 ! use oa_type
 ! interface interpolation
 !    module procedure interpolation_node
 !    module procedure interpolation_array
 ! end interface interpolation
contains
 !
 ! function interpolation_node(A, x, y, z) result(B)
 !   implicit none
 !   type(node) :: A
 !   type(node) :: B
 !   integer :: x
 !   integer, optional :: y, z
 !   type(node) :: ND, NA
 !   integer :: x1,y1,z1
 !
 !   x1 = x
 !
 !   print *,"interpolation_node"
 !
 !   if(present(y)) then
 !      y1 = y
 !   else
 !      y1 = 1
 !   end if
 !
 !   if(present(z)) then
 !      z1 = z
 !   else
 !      z1 = 1
 !   end if
 !   
 !   ND = new_local_int3([x1,y1,z1])
 !
 !   call c_new_node_op2(B%ptr, TYPE_INTERPOLATION, A%ptr, ND%ptr)    
 !
 !   call set_rvalue(B)
 !   call try_destroy(A)
 !   call destroy(NA)
 !   call destroy(ND)
 ! end function
 !
 ! function interpolation_array(A, x, y, z) result(B)
 !   implicit none
 !   type(array) :: A
 !   type(node) :: B
 !   integer :: x
 !   integer, optional :: y, z
 !   type(node) :: ND, NA
 !   integer :: x1,y1,z1
 !
 !   x1 = x
 !
 !   !print *,"interpolation_array"
 !
 !   if(present(y)) then
 !      y1 = y
 !   else
 !      y1 = 1
 !   end if
 !
 !   if(present(z)) then
 !      z1 = z
 !   else
 !      z1 = 1
 !   end if
 !   
 !   ND = new_local_int3([x1,y1,z1])
 !
 !   call c_new_node_array(NA%ptr, A%ptr)
 !   call c_new_node_op2(B%ptr, TYPE_INTERPOLATION, NA%ptr, ND%ptr)
 !
 !   call set_rvalue(B)
 !   call try_destroy(A)
 !   call destroy(NA)
 !   call destroy(ND)
 ! end function
end module
