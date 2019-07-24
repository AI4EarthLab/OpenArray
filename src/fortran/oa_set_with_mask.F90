
module oa_set_with_mask
  use iso_c_binding
  use oa_type
  use oa_utils
  
  interface set
     module procedure set_with_mask_array_const_int_array
     module procedure set_with_mask_array_const_int_node
     module procedure set_with_mask_array_const_float_array
     module procedure set_with_mask_array_const_float_node
     module procedure set_with_mask_array_const_double_array
     module procedure set_with_mask_array_const_double_node
     module procedure set_with_mask_array_array_array
     module procedure set_with_mask_array_node_array
     module procedure set_with_mask_array_array_node
     module procedure set_with_mask_array_node_node
  end interface set

contains

  subroutine set_with_mask_array_const_int_array(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_int_array &
            (A,  val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_int_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         integer, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, intent(in) :: val
    type(array), intent(in) :: mask
    call c_set_with_mask_array_const_int_array(A%ptr, &
         val, mask%ptr)

    call try_destroy(mask)
  end subroutine
  
  subroutine set_with_mask_array_const_int_node(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_int_array(A,&
            val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_int_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         integer, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(in) :: A
    integer, intent(in) :: val
    type(node), intent(in) :: mask
    type(array) :: mask_array, mask1

    call eval(mask_array, mask)
    
    call c_set_with_mask_array_const_int_array(A%ptr, &
         val, mask_array%ptr)

    call try_destroy(mask)
    call destroy(mask_array)
  end subroutine

  subroutine set_with_mask_array_const_float_array(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_float_array &
            (A,  val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_float_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, intent(in) :: val
    type(array), intent(in) :: mask
    call c_set_with_mask_array_const_float_array(A%ptr, &
         val, mask%ptr)

    call try_destroy(mask)
  end subroutine
  
  subroutine set_with_mask_array_const_float_node(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_float_array(A,&
            val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_float_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(in) :: A
    real, intent(in) :: val
    type(node), intent(in) :: mask
    type(array) :: mask_array, mask1

    call eval(mask_array, mask)
    
    call c_set_with_mask_array_const_float_array(A%ptr, &
         val, mask_array%ptr)

    call try_destroy(mask)
    call destroy(mask_array)
  end subroutine

  subroutine set_with_mask_array_const_double_array(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_double_array &
            (A,  val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_double_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real(8), value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), intent(in) :: val
    type(array), intent(in) :: mask
    call c_set_with_mask_array_const_double_array(A%ptr, &
         val, mask%ptr)

    call try_destroy(mask)
  end subroutine
  
  subroutine set_with_mask_array_const_double_node(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_double_array(A,&
            val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_double_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real(8), value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(in) :: A
    real(8), intent(in) :: val
    type(node), intent(in) :: mask
    type(array) :: mask_array, mask1

    call eval(mask_array, mask)
    
    call c_set_with_mask_array_const_double_array(A%ptr, &
         val, mask_array%ptr)

    call try_destroy(mask)
    call destroy(mask_array)
  end subroutine



  subroutine set_with_mask_array_array_array(A, B, mask)
     implicit none
     
     interface
        subroutine c_set_with_mask_array_array_array(a, b, mask) &
             bind(C, name="c_set_with_mask_array_array_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
          type(c_ptr) :: mask
        end subroutine
     end interface
     
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(array), intent(in) :: mask
    
    call c_set_with_mask_array_array_array(A%ptr, B%ptr, mask%ptr)

    call try_destroy(B)
    call try_destroy(mask)
  end subroutine
  
  subroutine set_with_mask_array_node_array(A, B, mask)
    implicit none
    interface
        subroutine c_set_with_mask_array_array_array(a, b, mask) &
             bind(C, name="c_set_with_mask_array_array_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
          type(c_ptr) :: mask
        end subroutine
     end interface

    type(array), intent(in) :: A
    type(node),  intent(in) :: B
    type(array) :: B_array
    type(array), intent(in) :: mask

    call eval(B_array, B)

    call c_set_with_mask_array_array_array(A%ptr, B_array%ptr, &
         mask%ptr)

    call try_destroy(B)
    call try_destroy(mask)
    call destroy(B_array)
  end subroutine

 
  subroutine set_with_mask_array_array_node(A, B, mask)
     implicit none
     
     interface
        subroutine c_set_with_mask_array_array_array(a, b, mask) &
             bind(C, name="c_set_with_mask_array_array_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
          type(c_ptr) :: mask
        end subroutine
     end interface
     
    type(array), intent(in) :: A
    type(array), intent(in) :: B
    type(node), intent(in) :: mask
    
    type(array) :: mask_array

    call eval(mask_array, mask)

    call c_set_with_mask_array_array_array(A%ptr, B%ptr, &
         mask_array%ptr)

    call try_destroy(B)
    call try_destroy(mask)
    call destroy(mask_array)

  end subroutine

  subroutine set_with_mask_array_node_node(A, B, mask)
     implicit none
     
     interface
        subroutine c_set_with_mask_array_node_node(a) &
             bind(C, name="c_set_with_mask_array_node_node")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a
        end subroutine
     end interface

     interface
        subroutine c_new_type_set(id1, id2) &
             bind(C, name="c_new_type_set")
          use iso_c_binding
          implicit none
         integer :: id1, id2
        end subroutine
     end interface

    
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node), intent(in) :: mask

    call c_new_type_set(B%id, mask%id)
    call c_set_with_mask_array_node_node(A%ptr)
    
  end subroutine

end module oa_set_with_mask
