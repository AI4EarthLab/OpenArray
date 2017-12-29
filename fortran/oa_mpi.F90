module oa_mpi
  use iso_c_binding

  interface
     subroutine c_get_rank(rank) &
          bind(C, name = 'c_get_rank')
      use iso_c_binding
      integer(c_int), intent(out) :: rank
    end subroutine
  end interface

  interface
     subroutine c_get_size(rank) &
          bind(C, name = 'c_get_size')
      use iso_c_binding
      integer(c_int), intent(out) :: rank
    end subroutine
  end interface
  
  ! interface
  !   subroutine c_get_size(size, fcomm) &
  !     bind(C, name = 'c_get_size')
  !     use iso_c_binding
  !     integer(c_int) :: size
  !     integer(c_int), intent(in), VALUE :: fcomm
  !   end subroutine
  ! end interface
  
contains
  
  function get_rank() result(rank)
    implicit none
    integer :: rank

    call c_get_rank(rank)
  end function

  function get_size() result(size)
    implicit none
    integer :: size

    call c_get_rank(size)
  end function
  
end module
