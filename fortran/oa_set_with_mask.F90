///:mute
///:set types = &
     [['int',   'integer'], &
     [ 'float',  'real'], &
     [ 'double','real(8)']]
///:endmute

module oa_set_with_mask
  use iso_c_binding
  use oa_type
  use oa_utils
  
  interface set
     ///:for type1 in types
     module procedure set_with_mask_array_const_${type1[0]}$_array
     module procedure set_with_mask_array_const_${type1[0]}$_node
     ///:endfor
     module procedure set_with_mask_array_array_array
     module procedure set_with_mask_array_node_array
     module procedure set_with_mask_array_array_node
     module procedure set_with_mask_array_node_node
  end interface set

contains

  ///:for type1 in types
  subroutine set_with_mask_array_const_${type1[0]}$_array(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_${type1[0]}$_array &
            (A,  val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_${type1[0]}$_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(inout) :: A
    ${type1[1]}$, intent(in) :: val
    type(array), intent(in) :: mask
    call c_set_with_mask_array_const_${type1[0]}$_array(A%ptr, &
         val, mask%ptr)

    call try_destroy(mask)
  end subroutine
  
  subroutine set_with_mask_array_const_${type1[0]}$_node(A, &
       val, mask)
    implicit none

    interface
       subroutine c_set_with_mask_array_const_${type1[0]}$_array(A,&
            val, mask) &
            bind(C, &
            name="c_set_with_mask_array_const_${type1[0]}$_array")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
         type(c_ptr) :: mask
       end subroutine
    end interface

    type(array), intent(in) :: A
    ${type1[1]}$, intent(in) :: val
    type(node), intent(in) :: mask
    type(array) :: mask_array, mask1

    call eval(mask_array, mask)
    
    call c_set_with_mask_array_const_${type1[0]}$_array(A%ptr, &
         val, mask_array%ptr)

    call try_destroy(mask)
    call destroy(mask_array)
  end subroutine

  ///:endfor


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
        subroutine c_set_with_mask_array_array_array(a, b, mask) &
             bind(C, name="c_set_with_mask_array_array_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
          type(c_ptr) :: mask
        end subroutine
     end interface
     
    type(array), intent(in) :: A
    type(node), intent(in) :: B
    type(node), intent(in) :: mask
    
    type(array) :: mask_array
    type(array) :: B_array

    call eval(mask_array, mask)
    call eval(B_array, B)

    call c_set_with_mask_array_array_array(A%ptr, B_array%ptr, &
         mask_array%ptr)

    call try_destroy(B)
    call try_destroy(mask)
    call destroy(mask_array)
    call destroy(B_array)
    
  end subroutine
end module oa_set_with_mask
