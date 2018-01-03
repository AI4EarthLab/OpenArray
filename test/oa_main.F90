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

  integer ierr, num_procs, my_id
  CHARACTER(len=32) :: arg
  integer di,dj,dk
  integer step
  integer s
  type(array) :: A, B, C
  real :: start, finish


  type(array) ::  q2f
  type(array) ::  q2f1
  type(array) ::  w
  type(array) ::  q2
  type(array) ::  dt_3d
  type(array) ::  u
  type(array) ::  aam
  type(array) ::  h_3d
  type(array) ::  q2b
  type(array) ::  dum_3d
  type(array) ::  v
  type(array) ::  dvm_3d
  call oa_init(MPI_COMM_WORLD, [-1,-1, -1])

  !call test_init(6, 6, 3, MPI_COMM_WORLD)
  call test_init(10, 10, 10, MPI_COMM_WORLD)

  step = 1

  !q2f1    = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !q2f     = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !!a1      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !!a2      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !!a3      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !!w       = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !w       = ones(m, n, k, dt = OA_FLOAT, sw=1)
  !q2      = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !dt_3d   = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !u       = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !aam     = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !h_3d    = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !q2b     = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !dum_3d  = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !v       = seqs(m, n, k, dt = OA_FLOAT, sw=1)
  !dvm_3d  = seqs(m, n, k, dt = OA_FLOAT, sw=1)

  q2f1    = rands(m, n, k, dt = OA_FLOAT, sw=1)
  q2f     = rands(m, n, k, dt = OA_FLOAT, sw=1)
  w       = rands(m, n, k, dt = OA_FLOAT, sw=1)
  q2      = rands(m, n, k, dt = OA_FLOAT, sw=1)
  dt_3d   = rands(m, n, k, dt = OA_FLOAT, sw=1)
  u       = rands(m, n, k, dt = OA_FLOAT, sw=1)
  aam     = rands(m, n, k, dt = OA_FLOAT, sw=1)
  h_3d    = rands(m, n, k, dt = OA_FLOAT, sw=1)
  q2b     = rands(m, n, k, dt = OA_FLOAT, sw=1)
  dum_3d  = rands(m, n, k, dt = OA_FLOAT, sw=1)
  v       = rands(m, n, k, dt = OA_FLOAT, sw=1)
  dvm_3d  = rands(m, n, k, dt = OA_FLOAT, sw=1)
  !FSET(q2f,a1+a2*u)

  call cpu_time(start)


  do s = 1 , step
    q2f1=DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d)
    FSET(q2f,DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d))
  end do


  call cpu_time(finish)
  call display(q2f1, "==============q2f1===============")
  call display(q2f, "==============q2f===============")
  print '(" ",f6.3,"")',finish-start


  call oa_finalize()

end program main
