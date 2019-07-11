module oa_option
#ifndef SUNWAY
  use oa_type
  
  interface oa_get_option
     module procedure oa_option_int_int
     module procedure oa_option_int_double
     module procedure oa_option_int_float
     module procedure oa_option_double_int
     module procedure oa_option_double_double
     module procedure oa_option_double_float
     module procedure oa_option_float_int
     module procedure oa_option_float_double
     module procedure oa_option_float_float
  end interface oa_get_option
  
contains
  
  subroutine oa_option_int_int(i, key, v)
    implicit none
    character(len=*) :: key
    integer :: i 
    integer :: v !v is default value

    interface
       subroutine c_oa_option_int_int(i, key, v) &
            bind(C, name = "c_oa_option_int_int")
         use iso_c_binding
         implicit none
         integer(c_int) :: i                  
         character(c_char) :: key(*)
         integer(c_int), value :: v
       end subroutine
    end interface

    call c_oa_option_int_int(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_int_double(i, key, v)
    implicit none
    character(len=*) :: key
    integer :: i 
    real(kind=8) :: v !v is default value

    interface
       subroutine c_oa_option_int_double(i, key, v) &
            bind(C, name = "c_oa_option_int_double")
         use iso_c_binding
         implicit none
         integer(c_int) :: i                  
         character(c_char) :: key(*)
         real(kind=c_double), value :: v
       end subroutine
    end interface

    call c_oa_option_int_double(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_int_float(i, key, v)
    implicit none
    character(len=*) :: key
    integer :: i 
    real :: v !v is default value

    interface
       subroutine c_oa_option_int_float(i, key, v) &
            bind(C, name = "c_oa_option_int_float")
         use iso_c_binding
         implicit none
         integer(c_int) :: i                  
         character(c_char) :: key(*)
         real(kind=c_float), value :: v
       end subroutine
    end interface

    call c_oa_option_int_float(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_double_int(i, key, v)
    implicit none
    character(len=*) :: key
    real(kind=8) :: i 
    integer :: v !v is default value

    interface
       subroutine c_oa_option_double_int(i, key, v) &
            bind(C, name = "c_oa_option_double_int")
         use iso_c_binding
         implicit none
         real(kind=c_double) :: i                  
         character(c_char) :: key(*)
         integer(c_int), value :: v
       end subroutine
    end interface

    call c_oa_option_double_int(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_double_double(i, key, v)
    implicit none
    character(len=*) :: key
    real(kind=8) :: i 
    real(kind=8) :: v !v is default value

    interface
       subroutine c_oa_option_double_double(i, key, v) &
            bind(C, name = "c_oa_option_double_double")
         use iso_c_binding
         implicit none
         real(kind=c_double) :: i                  
         character(c_char) :: key(*)
         real(kind=c_double), value :: v
       end subroutine
    end interface

    call c_oa_option_double_double(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_double_float(i, key, v)
    implicit none
    character(len=*) :: key
    real(kind=8) :: i 
    real :: v !v is default value

    interface
       subroutine c_oa_option_double_float(i, key, v) &
            bind(C, name = "c_oa_option_double_float")
         use iso_c_binding
         implicit none
         real(kind=c_double) :: i                  
         character(c_char) :: key(*)
         real(kind=c_float), value :: v
       end subroutine
    end interface

    call c_oa_option_double_float(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_float_int(i, key, v)
    implicit none
    character(len=*) :: key
    real :: i 
    integer :: v !v is default value

    interface
       subroutine c_oa_option_float_int(i, key, v) &
            bind(C, name = "c_oa_option_float_int")
         use iso_c_binding
         implicit none
         real(kind=c_float) :: i                  
         character(c_char) :: key(*)
         integer(c_int), value :: v
       end subroutine
    end interface

    call c_oa_option_float_int(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_float_double(i, key, v)
    implicit none
    character(len=*) :: key
    real :: i 
    real(kind=8) :: v !v is default value

    interface
       subroutine c_oa_option_float_double(i, key, v) &
            bind(C, name = "c_oa_option_float_double")
         use iso_c_binding
         implicit none
         real(kind=c_float) :: i                  
         character(c_char) :: key(*)
         real(kind=c_double), value :: v
       end subroutine
    end interface

    call c_oa_option_float_double(i, string_f2c(key), v)
    
  end subroutine
  subroutine oa_option_float_float(i, key, v)
    implicit none
    character(len=*) :: key
    real :: i 
    real :: v !v is default value

    interface
       subroutine c_oa_option_float_float(i, key, v) &
            bind(C, name = "c_oa_option_float_float")
         use iso_c_binding
         implicit none
         real(kind=c_float) :: i                  
         character(c_char) :: key(*)
         real(kind=c_float), value :: v
       end subroutine
    end interface

    call c_oa_option_float_float(i, string_f2c(key), v)
    
  end subroutine
#endif
end module
