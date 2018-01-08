
module oa_utils
  use oa_type

  interface 
     subroutine usleep(useconds) bind(C)
       use iso_c_binding 
       implicit none
       integer(c_int32_t), value :: useconds
     end subroutine
  end interface

contains

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

end module oa_utils
