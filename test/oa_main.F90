#include "../fortran/config.h"
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
  integer :: step
  integer :: i
  !no split in z-direction
  call oa_init(MPI_COMM_WORLD, [-1, -1, 1])
!  call test_ones();
  !call oa_init(MPI_COMM_WORLD, [2, 2, 2])

!  call oa_get_option(step, "step", -1)
!  print*, "step = ", step
!
!  ! call test_memleak()
!
!  call test_init(4, 3, 2, MPI_COMM_WORLD)
!  
!  do i = 1, 1
!     print*, "i = ", i
!     !initialize the test module  
!
!     !call test_diag()
!     
!     ! call test_wave()
!
!     ! call test_grid1()
!     
!     ! call test_create_array()
!
!     ! call test_create_node()
!
!     ! call test_partition()
!
!     ! call test_basic()
!
!     ! call test_compare()
!
!     ! call test_math()
!
      call test_sub()
!
!     ! call test_sum()
!
!     ! call test_min_max()
!
!     ! call test_rep()
!
!     ! call test_operator()
!
!     ! call test_grid()
!
!     ! call test_shift()
!
!     ! call test_cache()
!
!     ! call test_set()
!
!     ! call test_pow()
!
!     ! call test_set_with_mask()
!
!     ! call test_io()
!
!     ! call test_get_ptr()
!
!     ! call test_tic_toc()
!
!     ! call test_fusion_operator()
!
!     ! call test_internal_q()
!
!     ! call test_pseudo()
!
!     ! call test_simple_stmt()
!     
!     ! call test_interpolation()
!
!     ! call test_mat_mult() 
!  end do
  !call wave(nt, nx, ny)
  !print*, 'wave: ',ft-st, ' seconds'

!  call cpu_time(st)
!  call Runge_Kutte(nt, nx, ny)
!  call cpu_time(ft)
!  print*, 'Runge_Kutte: ',ft-st, ' seconds'

!  call cpu_time(st)
!  call Euler(nt, nx, ny)
!  call cpu_time(ft)
!  print*, 'Euler: ',ft-st, ' seconds'

!  call cpu_time(st)
!  call height(nt, nx,ny)
!  call cpu_time(ft)
!  print*, 'height: ',ft-st, ' seconds'
 !call test_plus(1,2,2)
!call test_slice(1,2,2)
  call oa_finalize()

end program main
