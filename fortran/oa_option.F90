
module oa_option
  use oa_type
contains
  function oa_option_int(key, v) result(i)
    implicit none
    character(len=*) :: key
    integer :: v, i !v is default value

    interface
       subroutine c_oa_option_int(i, key, v) &
            bind(C, name = "c_oa_option_int")
         use iso_c_binding
         implicit none
         integer(c_int) :: i                  
         character(c_char) :: key(*)
         integer(c_int), value :: v
       end subroutine
    end interface

    call c_oa_option_int(i, string_f2c(key), v)
    
  end function
end module
