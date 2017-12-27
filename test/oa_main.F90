module test
  use iso_c_binding
  use oa_mod
  use oa_test
contains

end module

program main
  use mpi
  use oa_test
  implicit none

  !no split in z-direction
  call oa_init(MPI_COMM_WORLD, [-1,-1, 1])
 
  !initialize the test module  
  call test_init(6, 6, 6, MPI_COMM_WORLD) 
  
  call test_create_array()
  
  call test_create_node()

  call test_partition()

  call test_basic()

  call test_compare()

  call test_math()

  call test_sub()

  call test_sum()

  call test_rep()

  call test_operator()

  call test_grid()

  call test_shift()

  call test_cache()
  
  call test_set()

  call test_io()
  
  call oa_finalize()
  
end program main
