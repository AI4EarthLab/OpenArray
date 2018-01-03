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
  call oa_init(MPI_COMM_WORLD, [-1,-1, 1])

  call test_init(6, 6, 6, MPI_COMM_WORLD)

  step = 1

  q2f     = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !a1      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !a2      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  !a3      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  w       = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  q2      = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  dt_3d   = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  u       = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  aam     = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  h_3d    = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  q2b     = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  dum_3d  = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  v       = seqs(m, n, k, dt = OA_FLOAT, sw=2)
  dvm_3d  = seqs(m, n, k, dt = OA_FLOAT, sw=2)

  !FSET(q2f,a1+a2*u)

  call cpu_time(start)


  do s = 1 , step
    !q2f=DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d)
    FSET(q2f,DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d))
  end do


  call cpu_time(finish)
  call display(q2f, "==============q2f===============")
  print '(" ",f6.3,"")',finish-start


  call oa_finalize()

end program main