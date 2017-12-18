///:mute
///:set types = &
     [['int',   'integer'], &
     [ 'float',  'real'], &
     [ 'double','real(8)']]
///:endmute

module oa_set
  use iso_c_binding
  use oa_type

  interface set
     ///:for type1 in types
     module procedure set_ref_const_${type1[0]}$
     module procedure set_array_const_${type1[0]}$
     ///:endfor
     module procedure set_ref_array
     module procedure set_ref_ref

  end interface set

  interface assignment(=)
     ///:for type1 in types
     module procedure set_array_const_${type1[0]}$
     ///:endfor
  end interface assignment(=)
  
contains

  ///:for type1 in types
  subroutine set_ref_const_${type1[0]}$(A, val)
    implicit none

    interface
       subroutine c_set_ref_const_${type1[0]}$(A,  val) &
            bind(C, name="c_set_ref_const_${type1[0]}$")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
       end subroutine
    end interface

    type(node), intent(in) :: A
    ${type1[1]}$ :: val

    call c_set_ref_const_${type1[0]}$(A%ptr, val)

  end subroutine


  subroutine set_array_const_${type1[0]}$(A, val)
    implicit none

    interface
       subroutine c_set_array_const_${type1[0]}$(A,  val) &
            bind(C, name="c_set_array_const_${type1[0]}$")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
       end subroutine
    end interface

    type(array), intent(inout) :: A
    ${type1[1]}$, intent(in) :: val

    call c_set_array_const_${type1[0]}$(A%ptr, val)

  end subroutine
  ///:endfor


   subroutine set_ref_array(A, B)
     implicit none
     
     interface
        subroutine c_set_ref_array(a, b) &
             bind(C, name="c_set_ref_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
        end subroutine
     end interface
     
    type(node), intent(in) :: A
    type(array) :: B
    
    call c_set_ref_array(A%ptr, B%ptr)
    
  end subroutine
  
  subroutine set_ref_ref(A, B)
    implicit none
    interface
        subroutine c_set_ref_ref(a, b) &
             bind(C, name="c_set_ref_ref")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
        end subroutine
     end interface

    type(node), intent(in) :: A, B

    call c_set_ref_ref(A%ptr, B%ptr)
             
  end subroutine



end module oa_set
