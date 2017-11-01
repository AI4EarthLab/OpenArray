       
module operators
  use instrict :: iso_c_binding
  use oa_type

  interface operator (+)
     module procedure ops_real8_plus_array
     module procedure ops_real8_plus_node
     module procedure ops_real_plus_array
     module procedure ops_real_plus_node
     module procedure ops_integer_plus_array
     module procedure ops_integer_plus_node
     module procedure ops_array_plus_real8
     module procedure ops_array_plus_real
     module procedure ops_array_plus_integer
     module procedure ops_array_plus_array
     module procedure ops_array_plus_node
     module procedure ops_node_plus_real8
     module procedure ops_node_plus_real
     module procedure ops_node_plus_integer
     module procedure ops_node_plus_array
     module procedure ops_node_plus_node
  end interface operator (+)
  
  interface operator (-)
     module procedure ops_real8_minus_array
     module procedure ops_real8_minus_node
     module procedure ops_real_minus_array
     module procedure ops_real_minus_node
     module procedure ops_integer_minus_array
     module procedure ops_integer_minus_node
     module procedure ops_array_minus_real8
     module procedure ops_array_minus_real
     module procedure ops_array_minus_integer
     module procedure ops_array_minus_array
     module procedure ops_array_minus_node
     module procedure ops_node_minus_real8
     module procedure ops_node_minus_real
     module procedure ops_node_minus_integer
     module procedure ops_node_minus_array
     module procedure ops_node_minus_node
  end interface operator (-)
  
  interface operator (*)
     module procedure ops_real8_mult_array
     module procedure ops_real8_mult_node
     module procedure ops_real_mult_array
     module procedure ops_real_mult_node
     module procedure ops_integer_mult_array
     module procedure ops_integer_mult_node
     module procedure ops_array_mult_real8
     module procedure ops_array_mult_real
     module procedure ops_array_mult_integer
     module procedure ops_array_mult_array
     module procedure ops_array_mult_node
     module procedure ops_node_mult_real8
     module procedure ops_node_mult_real
     module procedure ops_node_mult_integer
     module procedure ops_node_mult_array
     module procedure ops_node_mult_node
  end interface operator (*)
  
  interface operator (/)
     module procedure ops_real8_divd_array
     module procedure ops_real8_divd_node
     module procedure ops_real_divd_array
     module procedure ops_real_divd_node
     module procedure ops_integer_divd_array
     module procedure ops_integer_divd_node
     module procedure ops_array_divd_real8
     module procedure ops_array_divd_real
     module procedure ops_array_divd_integer
     module procedure ops_array_divd_array
     module procedure ops_array_divd_node
     module procedure ops_node_divd_real8
     module procedure ops_node_divd_real
     module procedure ops_node_divd_integer
     module procedure ops_node_divd_array
     module procedure ops_node_divd_node
  end interface operator (/)
  
  interface operator (>)
     module procedure ops_real8_gt_array
     module procedure ops_real8_gt_node
     module procedure ops_real_gt_array
     module procedure ops_real_gt_node
     module procedure ops_integer_gt_array
     module procedure ops_integer_gt_node
     module procedure ops_array_gt_real8
     module procedure ops_array_gt_real
     module procedure ops_array_gt_integer
     module procedure ops_array_gt_array
     module procedure ops_array_gt_node
     module procedure ops_node_gt_real8
     module procedure ops_node_gt_real
     module procedure ops_node_gt_integer
     module procedure ops_node_gt_array
     module procedure ops_node_gt_node
  end interface operator (>)
  
  interface operator (>=)
     module procedure ops_real8_ge_array
     module procedure ops_real8_ge_node
     module procedure ops_real_ge_array
     module procedure ops_real_ge_node
     module procedure ops_integer_ge_array
     module procedure ops_integer_ge_node
     module procedure ops_array_ge_real8
     module procedure ops_array_ge_real
     module procedure ops_array_ge_integer
     module procedure ops_array_ge_array
     module procedure ops_array_ge_node
     module procedure ops_node_ge_real8
     module procedure ops_node_ge_real
     module procedure ops_node_ge_integer
     module procedure ops_node_ge_array
     module procedure ops_node_ge_node
  end interface operator (>=)
  
  interface operator (<)
     module procedure ops_real8_lt_array
     module procedure ops_real8_lt_node
     module procedure ops_real_lt_array
     module procedure ops_real_lt_node
     module procedure ops_integer_lt_array
     module procedure ops_integer_lt_node
     module procedure ops_array_lt_real8
     module procedure ops_array_lt_real
     module procedure ops_array_lt_integer
     module procedure ops_array_lt_array
     module procedure ops_array_lt_node
     module procedure ops_node_lt_real8
     module procedure ops_node_lt_real
     module procedure ops_node_lt_integer
     module procedure ops_node_lt_array
     module procedure ops_node_lt_node
  end interface operator (<)
  
  interface operator (<=)
     module procedure ops_real8_le_array
     module procedure ops_real8_le_node
     module procedure ops_real_le_array
     module procedure ops_real_le_node
     module procedure ops_integer_le_array
     module procedure ops_integer_le_node
     module procedure ops_array_le_real8
     module procedure ops_array_le_real
     module procedure ops_array_le_integer
     module procedure ops_array_le_array
     module procedure ops_array_le_node
     module procedure ops_node_le_real8
     module procedure ops_node_le_real
     module procedure ops_node_le_integer
     module procedure ops_node_le_array
     module procedure ops_node_le_node
  end interface operator (<=)
  
  interface operator (==)
     module procedure ops_real8_eq_array
     module procedure ops_real8_eq_node
     module procedure ops_real_eq_array
     module procedure ops_real_eq_node
     module procedure ops_integer_eq_array
     module procedure ops_integer_eq_node
     module procedure ops_array_eq_real8
     module procedure ops_array_eq_real
     module procedure ops_array_eq_integer
     module procedure ops_array_eq_array
     module procedure ops_array_eq_node
     module procedure ops_node_eq_real8
     module procedure ops_node_eq_real
     module procedure ops_node_eq_integer
     module procedure ops_node_eq_array
     module procedure ops_node_eq_node
  end interface operator (==)
  
  interface operator (/=)
     module procedure ops_real8_ne_array
     module procedure ops_real8_ne_node
     module procedure ops_real_ne_array
     module procedure ops_real_ne_node
     module procedure ops_integer_ne_array
     module procedure ops_integer_ne_node
     module procedure ops_array_ne_real8
     module procedure ops_array_ne_real
     module procedure ops_array_ne_integer
     module procedure ops_array_ne_array
     module procedure ops_array_ne_node
     module procedure ops_node_ne_real8
     module procedure ops_node_ne_real
     module procedure ops_node_ne_integer
     module procedure ops_node_ne_array
     module procedure ops_node_ne_node
  end interface operator (/=)
  

  
contains

  !the following code using preprossor to create subroutines  
  function ops_real8_plus_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_real8_plus_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_real_plus_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_real_plus_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_integer_plus_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_integer_plus_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_array_plus_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_array_plus_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_array_plus_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_array_plus_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_array_plus_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_node_plus_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_node_plus_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_node_plus_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_node_plus_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_PLUS, C, D)

  end function

  function ops_node_plus_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_PLUS, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_minus_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_real8_minus_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_real_minus_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_real_minus_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_integer_minus_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_integer_minus_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_array_minus_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_array_minus_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_array_minus_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_array_minus_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_array_minus_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_node_minus_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_node_minus_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_node_minus_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_node_minus_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MINUS, C, D)

  end function

  function ops_node_minus_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_MINUS, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_mult_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_real8_mult_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_real_mult_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_real_mult_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_integer_mult_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_integer_mult_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_array_mult_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_array_mult_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_array_mult_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_array_mult_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_array_mult_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_node_mult_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_node_mult_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_node_mult_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_node_mult_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_MULT, C, D)

  end function

  function ops_node_mult_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_MULT, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_divd_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_real8_divd_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_real_divd_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_real_divd_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_integer_divd_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_integer_divd_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_array_divd_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_array_divd_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_array_divd_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_array_divd_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_array_divd_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_node_divd_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_node_divd_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_node_divd_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_node_divd_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_DIVD, C, D)

  end function

  function ops_node_divd_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_DIVD, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_gt_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_real8_gt_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_real_gt_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_real_gt_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_integer_gt_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_integer_gt_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_array_gt_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_array_gt_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_array_gt_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_array_gt_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_array_gt_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_node_gt_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_node_gt_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_node_gt_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_node_gt_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GT, C, D)

  end function

  function ops_node_gt_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_GT, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_ge_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_real8_ge_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_real_ge_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_real_ge_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_integer_ge_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_integer_ge_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_array_ge_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_array_ge_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_array_ge_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_array_ge_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_array_ge_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_node_ge_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_node_ge_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_node_ge_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_node_ge_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_GE, C, D)

  end function

  function ops_node_ge_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_GE, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_lt_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_real8_lt_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_real_lt_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_real_lt_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_integer_lt_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_integer_lt_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_array_lt_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_array_lt_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_array_lt_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_array_lt_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_array_lt_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_node_lt_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_node_lt_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_node_lt_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_node_lt_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LT, C, D)

  end function

  function ops_node_lt_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_LT, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_le_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_real8_le_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_real_le_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_real_le_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_integer_le_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_integer_le_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_array_le_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_array_le_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_array_le_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_array_le_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_array_le_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_node_le_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_node_le_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_node_le_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_node_le_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_LE, C, D)

  end function

  function ops_node_le_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_LE, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_eq_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_real8_eq_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_real_eq_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_real_eq_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_integer_eq_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_integer_eq_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_array_eq_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_array_eq_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_array_eq_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_array_eq_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_array_eq_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_node_eq_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_node_eq_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_node_eq_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_node_eq_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_EQ, C, D)

  end function

  function ops_node_eq_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_EQ, C, D)

  end function

  !the following code using preprossor to create subroutines  
  function ops_real8_ne_array(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_real8_ne_node(A, B) result(res)
    implicit none       
    real(8), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_real_ne_array(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_real_ne_node(A, B) result(res)
    implicit none       
    real, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_integer_ne_array(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_integer_ne_node(A, B) result(res)
    implicit none       
    integer, intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_array_ne_real8(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_array_ne_real(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_array_ne_integer(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_array_ne_array(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_array_ne_node(A, B) result(res)
    implicit none       
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    call create_node(C, A)

    D%ptr = B%ptr

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_node_ne_real8(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real(8), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_node_ne_real(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    real, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_node_ne_integer(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    integer, intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_node_ne_array(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(array), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    call create_node(D, B)

    call create_node(res, TYPE_NE, C, D)

  end function

  function ops_node_ne_node(A, B) result(res)
    implicit none       
    type(node), intent(in) :: A
    type(node), intent(in) :: B
    type(node) :: C, D
    type(node) :: res

    C%ptr = A%ptr

    D%ptr = B%ptr

    call create_node(res, TYPE_NE, C, D)

  end function


end module
