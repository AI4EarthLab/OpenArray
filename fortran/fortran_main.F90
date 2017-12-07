module test
  use iso_c_binding
  use oa_type
  use oa_utils
  use mpi
  use oa_test
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

    A = ones(4, 4, 1)
    call display_array(A%ptr)
 
    A = seqs(4,4,1,1,0)
    call display_array(A%ptr)
 
    A = rands(4,4,1)
    call display_array(A%ptr)

    B = zeros(4,4,1)
    call display_array(B%ptr)
    A = B
    call display_array(A%ptr)
    
    A = consts_int(4,4,1,1)
    call display_array(A%ptr)

    A = consts_float(4,4,1,2.0)
    call display_array(A%ptr)

    A = consts_double(4,4,1,3.0_8)
    call display_array(A%ptr)

  end subroutine

  subroutine test_create_node()
    implicit none
    
    integer(c_int) :: m, n, k, rank, fcomm
    type(Array) :: A, B, C 
    type(Node) :: X, Y, Z

    m = 4
    n = 4
    k = 1

    fcomm = MPI_COMM_WORLD
    call get_rank(rank, fcomm)

    call c_new_seqs_scalar_node_int(X%ptr, 1, fcomm)
    call display_node(X%ptr)

    call c_new_seqs_scalar_node_float(Y%ptr, 2.1, fcomm)
    call display_node(Y%ptr)

    call c_new_seqs_scalar_node_double(Z%ptr, 3.1_8, fcomm)
    call display_node(Z%ptr)

  end subroutine

end module

program main
  use mpi
  use test

  implicit none
  
  call oa_mpi_init()
  
  !call test_create_array()
  !call test_create_node()

  !call oa_mpi_finalize()

  call test_parition()
end program main
