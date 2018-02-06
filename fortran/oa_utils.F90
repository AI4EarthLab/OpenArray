
module oa_utils

  interface 
     subroutine usleep(useconds) bind(C)
       use iso_c_binding 
       implicit none
       integer(c_int32_t), value :: useconds
     end subroutine
  end interface

contains

  subroutine print_c_ptr(arg)
    use iso_c_binding
    implicit none
    type(c_ptr), intent(in) :: arg
    interface
       subroutine c_print_c_ptr(arg) &
            bind(C, name="c_print_c_ptr")
         use iso_c_binding
         implicit none
         type(c_ptr) :: arg
       end subroutine
    end interface

    call c_print_c_ptr(arg)
    
  end subroutine

  
  !> convert a fortran string to C string
  function string_f2c(f_string) result(c_string)
    use iso_c_binding
    character(len=*):: f_string
    CHARACTER(LEN=LEN_TRIM(f_string)+1,KIND=C_CHAR) :: c_string

    c_string = trim(f_string) // C_NULL_CHAR
  end function

  !> convert an integer to string
  function i2s(i, frt) result(str)
    implicit none
    integer, intent(in) :: i
    character(len=200) :: str
    character(len=*), intent(in), optional :: frt
    character(len=*), parameter :: frt1='(I0.1)'

    if(present(frt)) then
       write(str, frt) i
    else
       write(str, frt1) i
    end if
  end function
  
  ///:for t in ['tic', 'toc']
  subroutine ${t}$(key)
    use iso_c_binding
    implicit none
    character(len=*) :: key

    interface
       subroutine c_${t}$(key) &
            bind(C, name="c_${t}$")
         use iso_c_binding
         implicit none
         character(kind=c_char) :: key(*)
       end subroutine
    end interface

    call c_${t}$(string_f2c(key))
  end subroutine
  ///:endfor

  subroutine show_timer()
    use iso_c_binding
    implicit none

    interface
       subroutine c_show_timer() &
            bind(C, name="c_show_timer")
         implicit none
       end subroutine
    end interface

    call c_show_timer()
  end subroutine

  subroutine open_debug()
    use iso_c_binding
    implicit none

    interface
      subroutine c_open_debug() &
          bind(C, name="c_open_debug")
        implicit none
      end subroutine
    end interface

    interface
       subroutine abort() bind(C, name="abort")
       end subroutine abort
    end interface

    call c_open_debug()
  end subroutine

  subroutine close_debug()
    use iso_c_binding
    implicit none

    interface
      subroutine c_close_debug() &
          bind(C, name="c_close_debug")
        implicit none
      end subroutine
    end interface

    call c_close_debug()
  end subroutine

  subroutine assert(condition, file, linenum, msg)
    implicit none
    logical :: condition
    integer, optional :: linenum
    character(len=*), optional :: msg
    character(len=*), optional :: file

    if(.not. condition) then
       write(*,"(A)", advance="no") &
            "Error: assertation failed"
       if(present(linenum)) then
          write(*,"(A, A, A, I0, A)", advance="no") &
               " in ", file, " at line ",linenum, '.'
       endif
       if(present(msg)) write(*,*) "Reason: ", msg
       call abort()
    endif
  end subroutine assert

end module oa_utils
