module oa_option
#ifndef SUNWAY
  use oa_type
  ///:mute
  ///:set TYPE = [['int','integer','integer(c_int)'], &
       ['double','real(kind=8)', 'real(kind=c_double)'], &
       ['float', 'real', 'real(kind=c_float)']]
  ///:endmute
  
  interface oa_get_option
     ///:for ti in TYPE
     ///:for tv in TYPE
     module procedure oa_option_${ti[0]}$_${tv[0]}$
     ///:endfor
     ///:endfor
  end interface oa_get_option
  
contains
  
  ///:for ti in TYPE
  ///:for tv in TYPE
  subroutine oa_option_${ti[0]}$_${tv[0]}$(i, key, v)
    implicit none
    character(len=*) :: key
    ${ti[1]}$ :: i 
    ${tv[1]}$ :: v !v is default value

    interface
       subroutine c_oa_option_${ti[0]}$_${tv[0]}$(i, key, v) &
            bind(C, name = "c_oa_option_${ti[0]}$_${tv[0]}$")
         use iso_c_binding
         implicit none
         ${ti[2]}$ :: i                  
         character(c_char) :: key(*)
         ${tv[2]}$, value :: v
       end subroutine
    end interface

    call c_oa_option_${ti[0]}$_${tv[0]}$(i, string_f2c(key), v)
    
  end subroutine
  ///:endfor
  ///:endfor
#endif
end module
