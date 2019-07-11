#include "config.h"
 !
module oa_mat_mult
 ! use iso_c_binding
 ! use oa_type
 ! interface mat_mult
 !    module procedure mat_mult_node_node
 !    module procedure mat_mult_array_node
 !    module procedure mat_mult_node_array
 !    module procedure mat_mult_array_array
 ! end interface mat_mult
contains
 ! 
 ! function mat_mult_node_node(A, B) result(res)
 !   implicit none
 !   type(node) :: A, B
 !   type(node) :: res
 !
 !   call c_new_node_op2(res%ptr, TYPE_MAT_MULT, A%ptr, B%ptr)    
 !
 !   call set_rvalue(res)
 !   call try_destroy(A)
 !   call try_destroy(B)
 ! end function
 !
 ! function mat_mult_array_node(A, B) result(res)
 !   implicit none
 !   type(array) :: A
 !   type(node) :: B
 !   type(node) :: res
 !   type(node) :: NA
 !
 !   call c_new_node_array(NA%ptr, A%ptr)
 !   call c_new_node_op2(res%ptr, TYPE_MAT_MULT, NA%ptr, B%ptr)    
 !
 !   call set_rvalue(res)
 !   call try_destroy(A)
 !   call try_destroy(B)
 !   call destroy(NA)
 ! end function
 !
 ! function mat_mult_node_array(A, B) result(res)
 !   implicit none
 !   type(node) :: A
 !   type(array) :: B
 !   type(node) :: res
 !   type(node) :: NB
 !
 !   call c_new_node_array(NB%ptr, B%ptr)
 !   call c_new_node_op2(res%ptr, TYPE_MAT_MULT, A%ptr, NB%ptr)    
 !
 !   call set_rvalue(res)
 !   call try_destroy(A)
 !   call try_destroy(B)
 !   call destroy(NB)
 ! end function
 !
 ! function mat_mult_array_array(A, B) result(res)
 !   implicit none
 !   type(array) :: A
 !   type(array) :: B
 !   type(node) :: res
 !   type(node) :: ND, NA
 !
 !   call c_new_node_array(NA%ptr, A%ptr)
 !   call c_new_node_array(ND%ptr, B%ptr)
 !   call c_new_node_op2(res%ptr, TYPE_MAT_MULT, NA%ptr, ND%ptr)    
 !
 !   call set_rvalue(res)
 !   call try_destroy(A)
 !   call try_destroy(B)
 !   call destroy(NA)
 !   call destroy(ND)
 ! end function
end module
