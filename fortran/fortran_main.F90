module test
  use iso_c_binding
  use oa_type
  use oa_utils
  use mpi

contains
  subroutine test_create_array()
    implicit none
    
    integer(c_int) :: m, n, k, rank, fcomm
    type(Array) :: A, B

    m = 4
    n = 4
    k = 1

    fcomm = MPI_COMM_WORLD
    call get_rank(rank, fcomm)

    !A = ones(4, 4, 1)
    !call display_array(A%ptr)
!   call display_array(A%ptr)
!   if (rank == 0) print *, rank, A%ptr

!   A = seqs(4,4,1,1,0)
!   call display_array(A%ptr)
!   if (rank == 0) print *, rank, A%ptr

!   A = rands(4,4,1)
!   call display_array(A%ptr)
!   if (rank == 0) print *, rank, A%ptr

    B = rands(4,4,1)
    call display_array(B%ptr)
    A = B
    call display_array(A%ptr)

  end subroutine
end module

program main
  use mpi
  use test

  implicit none
  
  call oa_mpi_init()
  
  call test_create_array()

  call oa_mpi_finalize()

end program main
