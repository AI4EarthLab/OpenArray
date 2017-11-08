module test
  use iso_c_binding
  use oa_type
  use oa_utils
contains
  subroutine test_ones()
    implicit none
    
    integer(c_int) :: m, n, k
    type(Array) :: A, B

    m = 4
    n = 4
    k = 1

    A = ones(4, 4, 1)
    call display_array(A%ptr)

  end subroutine
end module

program main
  use mpi
  use test

  implicit none
  
  call oa_mpi_init()
  
  call test_ones()

  call oa_mpi_finalize()

end program main
