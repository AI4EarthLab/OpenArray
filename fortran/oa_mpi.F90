module oa_mpi
  use iso_c_binding

  interface
    subroutine c_get_rank(rank, fcomm) bind(C, name = 'c_get_rank')
      use iso_c_binding
      integer(c_int) :: rank
      integer(c_int), intent(in), VALUE :: fcomm
    end subroutine
  end interface

  interface
    subroutine c_get_size(size, fcomm) bind(C, name = 'c_get_size')
      use iso_c_binding
      integer(c_int) :: size
      integer(c_int), intent(in), VALUE :: fcomm
    end subroutine
  end interface
  
contains
  
  function get_rank(fcomm) result(rank)
    integer(c_int), intent(in) :: fcomm
    integer :: rank

    call c_get_rank(rank, fcomm)
  end function

  function get_size(fcomm) result(rank)
    integer(c_int), intent(in) :: fcomm
    integer :: size

    call c_get_rank(size, fcomm)
  end function
  
end module
