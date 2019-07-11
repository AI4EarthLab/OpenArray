  
  



























































#include "../fortran/config.h"
module oa_test
  use oa_mod
  integer :: m, n, k
  integer(c_int) :: rank, fcomm



contains

  subroutine test_diag()
    implicit none
    type(array) :: A, B, C, dx, dy, dz, rho, rmean
    integer :: i, N

    ! N = 10000000

    dx = load("dx.nc", "data")
    dy = load("dy.nc", "data")
    dz = load("dz.nc", "data")
    rho = load("rho.nc", "data")
    rmean = load("rmean.nc", "data")

    call grid_init('C', dx, dy, dz)
    call grid_bind(rho, 3)
    call grid_bind(rmean, 3)

    call disp(rho, "rho = ")
    call disp(rmean, "rmean = ")    
    call disp(dz, "dz = ")
    call open_debug()
    A = DZB(rho - rmean)
    call close_debug()
    call disp(A, "A = ")
    ! A = load("el.nc", "data")
    ! call disp(A, "A = ")
    
    ! print*, has_nan_or_inf(A)
    ! call disp(C, "C = ")
  end subroutine


  subroutine test_memleak()
    implicit none
    type(array) :: A, B, C
    integer :: i, N

    N = 10000000
    
    ! do i = 1, N
    !    A = ones(100, 10, 10)
    !    call usleep(1000)
    ! end do


    do i = 1, N
       A = ones(10, 10, 10)
       B = ones(10, 10, 10)
       ! C = A
       ! C = B
       
       C = A + B

       call usleep(10000)
    end do

    call destroy(A)
    call destroy(B)
    call destroy(C)
    
    ! call disp(C, "C = ")
  end subroutine
  
  subroutine test_init(mm, nn, kk, comm)
    integer :: mm, nn, kk
    integer(c_int) :: comm
    
    m = mm
    n = nn
    k = kk

    fcomm = comm
    rank = get_rank()
  end subroutine

  subroutine test_create_array()
    implicit none
    type(Array) :: A, B


    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=0)
    call display(A, "ones(m, n, k, sw=0)")

    A = ones(m, n, k, dt=OA_INT)
    call display(A, "ones(m, n, k, dt=OA_INT)")

    A = ones(m, n, k, sw=0, dt=OA_INT)
    call display(A, "ones(m, n, k, sw=0, dt=OA_INT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=1)
    call display(A, "ones(m, n, k, sw=1)")

    A = ones(m, n, k, dt=OA_INT)
    call display(A, "ones(m, n, k, dt=OA_INT)")

    A = ones(m, n, k, sw=1, dt=OA_INT)
    call display(A, "ones(m, n, k, sw=1, dt=OA_INT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=2)
    call display(A, "ones(m, n, k, sw=2)")

    A = ones(m, n, k, dt=OA_INT)
    call display(A, "ones(m, n, k, dt=OA_INT)")

    A = ones(m, n, k, sw=2, dt=OA_INT)
    call display(A, "ones(m, n, k, sw=2, dt=OA_INT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=0)
    call display(A, "ones(m, n, k, sw=0)")

    A = ones(m, n, k, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, dt=OA_FLOAT)")

    A = ones(m, n, k, sw=0, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, sw=0, dt=OA_FLOAT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=1)
    call display(A, "ones(m, n, k, sw=1)")

    A = ones(m, n, k, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, dt=OA_FLOAT)")

    A = ones(m, n, k, sw=1, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, sw=1, dt=OA_FLOAT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=2)
    call display(A, "ones(m, n, k, sw=2)")

    A = ones(m, n, k, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, dt=OA_FLOAT)")

    A = ones(m, n, k, sw=2, dt=OA_FLOAT)
    call display(A, "ones(m, n, k, sw=2, dt=OA_FLOAT)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=0)
    call display(A, "ones(m, n, k, sw=0)")

    A = ones(m, n, k, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, dt=OA_DOUBLE)")

    A = ones(m, n, k, sw=0, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, sw=0, dt=OA_DOUBLE)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=1)
    call display(A, "ones(m, n, k, sw=1)")

    A = ones(m, n, k, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, dt=OA_DOUBLE)")

    A = ones(m, n, k, sw=1, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, sw=1, dt=OA_DOUBLE)")

    A = ones(m, n, k)
    call display(A, "A = ones(m, n, k)")

    A = ones(m, n, k, sw=2)
    call display(A, "ones(m, n, k, sw=2)")

    A = ones(m, n, k, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, dt=OA_DOUBLE)")

    A = ones(m, n, k, sw=2, dt=OA_DOUBLE)
    call display(A, "ones(m, n, k, sw=2, dt=OA_DOUBLE)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=0)
    call display(A, "zeros(m, n, k, sw=0)")

    A = zeros(m, n, k, dt=OA_INT)
    call display(A, "zeros(m, n, k, dt=OA_INT)")

    A = zeros(m, n, k, sw=0, dt=OA_INT)
    call display(A, "zeros(m, n, k, sw=0, dt=OA_INT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=1)
    call display(A, "zeros(m, n, k, sw=1)")

    A = zeros(m, n, k, dt=OA_INT)
    call display(A, "zeros(m, n, k, dt=OA_INT)")

    A = zeros(m, n, k, sw=1, dt=OA_INT)
    call display(A, "zeros(m, n, k, sw=1, dt=OA_INT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=2)
    call display(A, "zeros(m, n, k, sw=2)")

    A = zeros(m, n, k, dt=OA_INT)
    call display(A, "zeros(m, n, k, dt=OA_INT)")

    A = zeros(m, n, k, sw=2, dt=OA_INT)
    call display(A, "zeros(m, n, k, sw=2, dt=OA_INT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=0)
    call display(A, "zeros(m, n, k, sw=0)")

    A = zeros(m, n, k, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, dt=OA_FLOAT)")

    A = zeros(m, n, k, sw=0, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, sw=0, dt=OA_FLOAT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=1)
    call display(A, "zeros(m, n, k, sw=1)")

    A = zeros(m, n, k, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, dt=OA_FLOAT)")

    A = zeros(m, n, k, sw=1, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, sw=1, dt=OA_FLOAT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=2)
    call display(A, "zeros(m, n, k, sw=2)")

    A = zeros(m, n, k, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, dt=OA_FLOAT)")

    A = zeros(m, n, k, sw=2, dt=OA_FLOAT)
    call display(A, "zeros(m, n, k, sw=2, dt=OA_FLOAT)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=0)
    call display(A, "zeros(m, n, k, sw=0)")

    A = zeros(m, n, k, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, dt=OA_DOUBLE)")

    A = zeros(m, n, k, sw=0, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, sw=0, dt=OA_DOUBLE)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=1)
    call display(A, "zeros(m, n, k, sw=1)")

    A = zeros(m, n, k, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, dt=OA_DOUBLE)")

    A = zeros(m, n, k, sw=1, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, sw=1, dt=OA_DOUBLE)")

    A = zeros(m, n, k)
    call display(A, "A = zeros(m, n, k)")

    A = zeros(m, n, k, sw=2)
    call display(A, "zeros(m, n, k, sw=2)")

    A = zeros(m, n, k, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, dt=OA_DOUBLE)")

    A = zeros(m, n, k, sw=2, dt=OA_DOUBLE)
    call display(A, "zeros(m, n, k, sw=2, dt=OA_DOUBLE)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=0)
    call display(A, "seqs(m, n, k, sw=0)")

    A = seqs(m, n, k, dt=OA_INT)
    call display(A, "seqs(m, n, k, dt=OA_INT)")

    A = seqs(m, n, k, sw=0, dt=OA_INT)
    call display(A, "seqs(m, n, k, sw=0, dt=OA_INT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=1)
    call display(A, "seqs(m, n, k, sw=1)")

    A = seqs(m, n, k, dt=OA_INT)
    call display(A, "seqs(m, n, k, dt=OA_INT)")

    A = seqs(m, n, k, sw=1, dt=OA_INT)
    call display(A, "seqs(m, n, k, sw=1, dt=OA_INT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=2)
    call display(A, "seqs(m, n, k, sw=2)")

    A = seqs(m, n, k, dt=OA_INT)
    call display(A, "seqs(m, n, k, dt=OA_INT)")

    A = seqs(m, n, k, sw=2, dt=OA_INT)
    call display(A, "seqs(m, n, k, sw=2, dt=OA_INT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=0)
    call display(A, "seqs(m, n, k, sw=0)")

    A = seqs(m, n, k, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, dt=OA_FLOAT)")

    A = seqs(m, n, k, sw=0, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, sw=0, dt=OA_FLOAT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=1)
    call display(A, "seqs(m, n, k, sw=1)")

    A = seqs(m, n, k, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, dt=OA_FLOAT)")

    A = seqs(m, n, k, sw=1, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, sw=1, dt=OA_FLOAT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=2)
    call display(A, "seqs(m, n, k, sw=2)")

    A = seqs(m, n, k, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, dt=OA_FLOAT)")

    A = seqs(m, n, k, sw=2, dt=OA_FLOAT)
    call display(A, "seqs(m, n, k, sw=2, dt=OA_FLOAT)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=0)
    call display(A, "seqs(m, n, k, sw=0)")

    A = seqs(m, n, k, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, dt=OA_DOUBLE)")

    A = seqs(m, n, k, sw=0, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, sw=0, dt=OA_DOUBLE)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=1)
    call display(A, "seqs(m, n, k, sw=1)")

    A = seqs(m, n, k, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, dt=OA_DOUBLE)")

    A = seqs(m, n, k, sw=1, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, sw=1, dt=OA_DOUBLE)")

    A = seqs(m, n, k)
    call display(A, "A = seqs(m, n, k)")

    A = seqs(m, n, k, sw=2)
    call display(A, "seqs(m, n, k, sw=2)")

    A = seqs(m, n, k, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, dt=OA_DOUBLE)")

    A = seqs(m, n, k, sw=2, dt=OA_DOUBLE)
    call display(A, "seqs(m, n, k, sw=2, dt=OA_DOUBLE)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=0)
    call display(A, "rands(m, n, k, sw=0)")

    A = rands(m, n, k, dt=OA_INT)
    call display(A, "rands(m, n, k, dt=OA_INT)")

    A = rands(m, n, k, sw=0, dt=OA_INT)
    call display(A, "rands(m, n, k, sw=0, dt=OA_INT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=1)
    call display(A, "rands(m, n, k, sw=1)")

    A = rands(m, n, k, dt=OA_INT)
    call display(A, "rands(m, n, k, dt=OA_INT)")

    A = rands(m, n, k, sw=1, dt=OA_INT)
    call display(A, "rands(m, n, k, sw=1, dt=OA_INT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=2)
    call display(A, "rands(m, n, k, sw=2)")

    A = rands(m, n, k, dt=OA_INT)
    call display(A, "rands(m, n, k, dt=OA_INT)")

    A = rands(m, n, k, sw=2, dt=OA_INT)
    call display(A, "rands(m, n, k, sw=2, dt=OA_INT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=0)
    call display(A, "rands(m, n, k, sw=0)")

    A = rands(m, n, k, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, dt=OA_FLOAT)")

    A = rands(m, n, k, sw=0, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, sw=0, dt=OA_FLOAT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=1)
    call display(A, "rands(m, n, k, sw=1)")

    A = rands(m, n, k, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, dt=OA_FLOAT)")

    A = rands(m, n, k, sw=1, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, sw=1, dt=OA_FLOAT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=2)
    call display(A, "rands(m, n, k, sw=2)")

    A = rands(m, n, k, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, dt=OA_FLOAT)")

    A = rands(m, n, k, sw=2, dt=OA_FLOAT)
    call display(A, "rands(m, n, k, sw=2, dt=OA_FLOAT)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=0)
    call display(A, "rands(m, n, k, sw=0)")

    A = rands(m, n, k, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, dt=OA_DOUBLE)")

    A = rands(m, n, k, sw=0, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, sw=0, dt=OA_DOUBLE)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=1)
    call display(A, "rands(m, n, k, sw=1)")

    A = rands(m, n, k, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, dt=OA_DOUBLE)")

    A = rands(m, n, k, sw=1, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, sw=1, dt=OA_DOUBLE)")

    A = rands(m, n, k)
    call display(A, "A = rands(m, n, k)")

    A = rands(m, n, k, sw=2)
    call display(A, "rands(m, n, k, sw=2)")

    A = rands(m, n, k, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, dt=OA_DOUBLE)")

    A = rands(m, n, k, sw=2, dt=OA_DOUBLE)
    call display(A, "rands(m, n, k, sw=2, dt=OA_DOUBLE)")

    ! A = ones(4, 4, 1)
    ! call display(A, "A")

    ! A = seqs(4,4,1,1,0)
    ! call display(A, "A")

    ! A = rands(4,4,1)
    ! call display(A, "A")

    ! B = zeros(4,4,1)
    ! call display(B, "B")

    ! A = B
    ! call display(A, "A")

    A = consts(4,4,1,1)
    call display(A, "A")

    A = consts(4,4,1,2.0)
    call display(A, "A")

    A = consts(4,4,1,3.0_8)
    call display(A, "A")

  end subroutine

  subroutine test_create_node()
    implicit none
    type(Array) :: A, B, C 
    type(Node) :: X, Y, Z

    call c_new_seqs_scalar_node_int(X%ptr, 1)
    call display(X, "X")

    call c_new_seqs_scalar_node_float(Y%ptr, 2.1)
    call display(Y, "Y")

    call c_new_seqs_scalar_node_double(Z%ptr, 3.1_8)
    call display(Z, "Z")

  end subroutine


  subroutine test_basic()
    type(array) :: A, B, C, D, E
    integer :: si
    real :: sr
    real(kind=8) :: sr8

    A = ones(m, n, k)
    call display(A, "A")

    B = seqs(m,n,k,1,0) + 1
    call display(B, "B")

    !array with array
    C = A + B
    call display(C, "A + B = ")
    C = A - B
    call display(C, "A - B = ")
    C = A * B
    call display(C, "A * B = ")
    C = A / B
    call display(C, "A / B = ")

    si  = 2;
    sr  = 3.0;
    sr8 = 4.0_8;

    !scalar with array
    C = si + B
    call display(C, "2 + B = ")
    C = sr + B
    call display(C, "3.0 + B = ")
    C = sr8 + B
    call display(C, "4.0_8 + B = ")
    C = si - B
    call display(C, "2 - B = ")
    C = sr - B
    call display(C, "3.0 - B = ")
    C = sr8 - B
    call display(C, "4.0_8 - B = ")
    C = si * B
    call display(C, "2 * B = ")
    C = sr * B
    call display(C, "3.0 * B = ")
    C = sr8 * B
    call display(C, "4.0_8 * B = ")
    C = si / B
    call display(C, "2 / B = ")
    C = sr / B
    call display(C, "3.0 / B = ")
    C = sr8 / B
    call display(C, "4.0_8 / B = ")

    !array with scalar
    C = B + si
    call display(C, "B + 2 = ")
    C = B + sr
    call display(C, "B + 3.0 = ")
    C = B + sr8
    call display(C, "B + 4.0_8 = ")
    C = B - si
    call display(C, "B - 2 = ")
    C = B - sr
    call display(C, "B - 3.0 = ")
    C = B - sr8
    call display(C, "B - 4.0_8 = ")
    C = B * si
    call display(C, "B * 2 = ")
    C = B * sr
    call display(C, "B * 3.0 = ")
    C = B * sr8
    call display(C, "B * 4.0_8 = ")
    C = B / si
    call display(C, "B / 2 = ")
    C = B / sr
    call display(C, "B / 3.0 = ")
    C = B / sr8
    call display(C, "B / 4.0_8 = ")

  end subroutine

    subroutine test_pow()
      implicit none
      type(array) :: A, B

      A = seqs(m, n, k, dt=OA_INT)

      B = A**(0.5)
      call display(B, "A**(0.5) = ")

      B = A**(0.5_8)
      call display(B, "A**(0.5_8) = ")
      
      B = A**2
      call display(B, "A**(2) = ")

      B = A**0
      call display(B, "A**(0) = ")
      A = seqs(m, n, k, dt=OA_FLOAT)

      B = A**(0.5)
      call display(B, "A**(0.5) = ")

      B = A**(0.5_8)
      call display(B, "A**(0.5_8) = ")
      
      B = A**2
      call display(B, "A**(2) = ")

      B = A**0
      call display(B, "A**(0) = ")
      A = seqs(m, n, k, dt=OA_DOUBLE)

      B = A**(0.5)
      call display(B, "A**(0.5) = ")

      B = A**(0.5_8)
      call display(B, "A**(0.5_8) = ")
      
      B = A**2
      call display(B, "A**(2) = ")

      B = A**0
      call display(B, "A**(0) = ")
      
    end subroutine
    
    subroutine test_logic()
      implicit none
      type(array) :: A, B, C, D
      
      A = rands(m, n, k)
      B = A > 0.2
      C = A < 0.7

      call display(B, "B = ")
      call display(C, "C = ")
      
      D = B .and. C
      call display(D, 'B .and. C = ')

      call display(A, "A = ")
      D = (A > 0.8) .or. (A < 0.2)
      call display(D, '(A > 0.8 .or A < 0.2) = ')
      
    end subroutine
    

  subroutine test_compare()
    type(array) :: A, B, C
    integer :: si
    real :: sr
    real(kind=8) :: sr8

    A = ones(4, 4, 1) + 2
    call display(A, "A")

    B = seqs(4,4,1,1,0) + 1
    call display(B, "B")


    !array with array
    C = A > B
    call display(C, "A > B = ")
    C = A >= B
    call display(C, "A >= B = ")
    C = A < B
    call display(C, "A < B = ")
    C = A <= B
    call display(C, "A <= B = ")
    C = A == B
    call display(C, "A == B = ")
    C = A /= B
    call display(C, "A /= B = ")

    si  = 2;
    sr  = 3.0;
    sr8 = 4.0_8;

    !scalar with array
    C = si > B
    call display(C, "2 > B = ")
    C = sr > B
    call display(C, "3.0 > B = ")
    C = sr8 > B
    call display(C, "4.0_8 > B = ")
    C = si >= B
    call display(C, "2 >= B = ")
    C = sr >= B
    call display(C, "3.0 >= B = ")
    C = sr8 >= B
    call display(C, "4.0_8 >= B = ")
    C = si < B
    call display(C, "2 < B = ")
    C = sr < B
    call display(C, "3.0 < B = ")
    C = sr8 < B
    call display(C, "4.0_8 < B = ")
    C = si <= B
    call display(C, "2 <= B = ")
    C = sr <= B
    call display(C, "3.0 <= B = ")
    C = sr8 <= B
    call display(C, "4.0_8 <= B = ")
    C = si == B
    call display(C, "2 == B = ")
    C = sr == B
    call display(C, "3.0 == B = ")
    C = sr8 == B
    call display(C, "4.0_8 == B = ")
    C = si /= B
    call display(C, "2 /= B = ")
    C = sr /= B
    call display(C, "3.0 /= B = ")
    C = sr8 /= B
    call display(C, "4.0_8 /= B = ")

    !array with scalar
    C = B > si
    call display(C, "B > 2 = ")
    C = B > sr
    call display(C, "B > 3.0 = ")
    C = B > sr8
    call display(C, "B > 4.0_8 = ")
    C = B >= si
    call display(C, "B >= 2 = ")
    C = B >= sr
    call display(C, "B >= 3.0 = ")
    C = B >= sr8
    call display(C, "B >= 4.0_8 = ")
    C = B < si
    call display(C, "B < 2 = ")
    C = B < sr
    call display(C, "B < 3.0 = ")
    C = B < sr8
    call display(C, "B < 4.0_8 = ")
    C = B <= si
    call display(C, "B <= 2 = ")
    C = B <= sr
    call display(C, "B <= 3.0 = ")
    C = B <= sr8
    call display(C, "B <= 4.0_8 = ")
    C = B == si
    call display(C, "B == 2 = ")
    C = B == sr
    call display(C, "B == 3.0 = ")
    C = B == sr8
    call display(C, "B == 4.0_8 = ")
    C = B /= si
    call display(C, "B /= 2 = ")
    C = B /= sr
    call display(C, "B /= 3.0 = ")
    C = B /= sr8
    call display(C, "B /= 4.0_8 = ")

  end subroutine


  subroutine test_math()
    type(array) :: A, B, C, D
    integer :: i

    A = ones(m, n, k, dt=OA_DOUBLE)
    B = ones(m, n, k, dt=OA_DOUBLE)
    C = ones(m, n, k, dt=OA_DOUBLE)
    D = seqs(m, n, k, dt=OA_DOUBLE)


    D = exp(C)

    ! call display(D, "exp(C) = ")

    D = exp(C*0.5)
    ! call display(D, "exp(C*0.5) = ")
    

    D = sin(C)

    ! call display(D, "sin(C) = ")

    D = sin(C*0.5)
    ! call display(D, "sin(C*0.5) = ")
    

    D = tan(C)

    ! call display(D, "tan(C) = ")

    D = tan(C*0.5)
    ! call display(D, "tan(C*0.5) = ")
    

    D = cos(C)

    ! call display(D, "cos(C) = ")

    D = cos(C*0.5)
    ! call display(D, "cos(C*0.5) = ")
    

    D = rcp(C)

    ! call display(D, "rcp(C) = ")

    

    D = sqrt(C)

    ! call display(D, "sqrt(C) = ")

    D = sqrt(C*0.5)
    ! call display(D, "sqrt(C*0.5) = ")
    

    D = asin(C)

    ! call display(D, "asin(C) = ")

    D = asin(C*0.5)
    ! call display(D, "asin(C*0.5) = ")
    

    D = acos(C)

    ! call display(D, "acos(C) = ")

    D = acos(C*0.5)
    ! call display(D, "acos(C*0.5) = ")
    

    D = atan(C)

    ! call display(D, "atan(C) = ")

    D = atan(C*0.5)
    ! call display(D, "atan(C*0.5) = ")
    

    D = abs(C)

    ! call display(D, "abs(C) = ")

    D = abs(C*0.5)
    ! call display(D, "abs(C*0.5) = ")
    

    D = log(C)

    ! call display(D, "log(C) = ")

    D = log(C*0.5)
    ! call display(D, "log(C*0.5) = ")
    

    D = +(C)

    ! call display(D, "+(C) = ")

    D = +(C*0.5)
    ! call display(D, "+(C*0.5) = ")
    

    D = -(C)

    ! call display(D, "-(C) = ")

    D = -(C*0.5)
    ! call display(D, "-(C*0.5) = ")
    

    D = log10(C)

    ! call display(D, "log10(C) = ")

    D = log10(C*0.5)
    ! call display(D, "log10(C*0.5) = ")
    

    D = tanh(C)

    ! call display(D, "tanh(C) = ")

    D = tanh(C*0.5)
    ! call display(D, "tanh(C*0.5) = ")
    

    D = sinh(C)

    ! call display(D, "sinh(C) = ")

    D = sinh(C*0.5)
    ! call display(D, "sinh(C*0.5) = ")
    

    D = cosh(C)

    ! call display(D, "cosh(C) = ")

    D = cosh(C*0.5)
    ! call display(D, "cosh(C*0.5) = ")
    

    D = seqs(m, n, k, dt=OA_DOUBLE)
    
    do i = 0, 100
       A = B + C + D
      ! FSET(A, B + C + D)
    end do

    call disp(A, 'A = ')
    
    call destroy(A)
    call destroy(B)
    call destroy(C)
    call destroy(D)
  end subroutine

  subroutine test_sub()
    implicit none
    type(array) :: A, B, C

    A = seqs(10, 8, 2);
    call display(A, "A = ")

    !sub index is from 1 
    B = sub(A, [1, 1], &
      [1,1],[1,1])

    call display(B, "sub(A, [1, 1], &
      &[1,1],[1,1]) = ")    
    B = sub(A, [2, 4], &
      [1,1],[1,1])

    call display(B, "sub(A, [2, 4], &
      &[1,1],[1,1]) = ")    
    B = sub(A, [6, 10], &
      [1,1],[1,1])

    call display(B, "sub(A, [6, 10], &
      &[1,1],[1,1]) = ")    
    B = sub(A, [1, 1], &
      [1,3],[1,1])

    call display(B, "sub(A, [1, 1], &
      &[1,3],[1,1]) = ")    
    B = sub(A, [2, 4], &
      [1,3],[1,1])

    call display(B, "sub(A, [2, 4], &
      &[1,3],[1,1]) = ")    
    B = sub(A, [6, 10], &
      [1,3],[1,1])

    call display(B, "sub(A, [6, 10], &
      &[1,3],[1,1]) = ")    
    B = sub(A, [1, 1], &
      [2,5],[1,1])

    call display(B, "sub(A, [1, 1], &
      &[2,5],[1,1]) = ")    
    B = sub(A, [2, 4], &
      [2,5],[1,1])

    call display(B, "sub(A, [2, 4], &
      &[2,5],[1,1]) = ")    
    B = sub(A, [6, 10], &
      [2,5],[1,1])

    call display(B, "sub(A, [6, 10], &
      &[2,5],[1,1]) = ")    
    B = sub(A, [1, 1], &
      [1,1],[1,2])

    call display(B, "sub(A, [1, 1], &
      &[1,1],[1,2]) = ")    
    B = sub(A, [2, 4], &
      [1,1],[1,2])

    call display(B, "sub(A, [2, 4], &
      &[1,1],[1,2]) = ")    
    B = sub(A, [6, 10], &
      [1,1],[1,2])

    call display(B, "sub(A, [6, 10], &
      &[1,1],[1,2]) = ")    
    B = sub(A, [1, 1], &
      [1,3],[1,2])

    call display(B, "sub(A, [1, 1], &
      &[1,3],[1,2]) = ")    
    B = sub(A, [2, 4], &
      [1,3],[1,2])

    call display(B, "sub(A, [2, 4], &
      &[1,3],[1,2]) = ")    
    B = sub(A, [6, 10], &
      [1,3],[1,2])

    call display(B, "sub(A, [6, 10], &
      &[1,3],[1,2]) = ")    
    B = sub(A, [1, 1], &
      [2,5],[1,2])

    call display(B, "sub(A, [1, 1], &
      &[2,5],[1,2]) = ")    
    B = sub(A, [2, 4], &
      [2,5],[1,2])

    call display(B, "sub(A, [2, 4], &
      &[2,5],[1,2]) = ")    
    B = sub(A, [6, 10], &
      [2,5],[1,2])

    call display(B, "sub(A, [6, 10], &
      &[2,5],[1,2]) = ")    

    B = sub(A, [1,3],[1,8],[1,2])
    call display(B, "sub(A, [1,3],[1,8],[1,2]) = ")

    B = sub(A, [1,3])
    call display(B, "sub(A, [1,3]) = ")

    B = sub(A, 1);
    call display(B, "sub(A, 1) = ")

    B = sub(A, 1, 2, 2);
    call display(B, "sub(A, 1, 2, 2) = ")


  end subroutine

  subroutine test_rep()
    implicit none
    type(array) :: A, B, C
    type(node) :: nn

    A = seqs(m, n, k)


    B = rep(A, 1, 1, 1)
    call display(B, "rep(A, 1, 1, 1) = ")

    C = rep(A+A, 1, 1, 1)
    call display(C, "rep(A+A, 1, 1, 1) = ")


    B = rep(A, 1, 1, 2)
    call display(B, "rep(A, 1, 1, 2) = ")

    C = rep(A+A, 1, 1, 2)
    call display(C, "rep(A+A, 1, 1, 2) = ")


    B = rep(A, 1, 2, 1)
    call display(B, "rep(A, 1, 2, 1) = ")

    C = rep(A+A, 1, 2, 1)
    call display(C, "rep(A+A, 1, 2, 1) = ")


    B = rep(A, 1, 2, 2)
    call display(B, "rep(A, 1, 2, 2) = ")

    C = rep(A+A, 1, 2, 2)
    call display(C, "rep(A+A, 1, 2, 2) = ")


    B = rep(A, 1, 3, 1)
    call display(B, "rep(A, 1, 3, 1) = ")

    C = rep(A+A, 1, 3, 1)
    call display(C, "rep(A+A, 1, 3, 1) = ")


    B = rep(A, 1, 3, 2)
    call display(B, "rep(A, 1, 3, 2) = ")

    C = rep(A+A, 1, 3, 2)
    call display(C, "rep(A+A, 1, 3, 2) = ")


    B = rep(A, 2, 1, 1)
    call display(B, "rep(A, 2, 1, 1) = ")

    C = rep(A+A, 2, 1, 1)
    call display(C, "rep(A+A, 2, 1, 1) = ")


    B = rep(A, 2, 1, 2)
    call display(B, "rep(A, 2, 1, 2) = ")

    C = rep(A+A, 2, 1, 2)
    call display(C, "rep(A+A, 2, 1, 2) = ")


    B = rep(A, 2, 2, 1)
    call display(B, "rep(A, 2, 2, 1) = ")

    C = rep(A+A, 2, 2, 1)
    call display(C, "rep(A+A, 2, 2, 1) = ")


    B = rep(A, 2, 2, 2)
    call display(B, "rep(A, 2, 2, 2) = ")

    C = rep(A+A, 2, 2, 2)
    call display(C, "rep(A+A, 2, 2, 2) = ")


    B = rep(A, 2, 3, 1)
    call display(B, "rep(A, 2, 3, 1) = ")

    C = rep(A+A, 2, 3, 1)
    call display(C, "rep(A+A, 2, 3, 1) = ")


    B = rep(A, 2, 3, 2)
    call display(B, "rep(A, 2, 3, 2) = ")

    C = rep(A+A, 2, 3, 2)
    call display(C, "rep(A+A, 2, 3, 2) = ")


    B = rep(A, 3, 1, 1)
    call display(B, "rep(A, 3, 1, 1) = ")

    C = rep(A+A, 3, 1, 1)
    call display(C, "rep(A+A, 3, 1, 1) = ")


    B = rep(A, 3, 1, 2)
    call display(B, "rep(A, 3, 1, 2) = ")

    C = rep(A+A, 3, 1, 2)
    call display(C, "rep(A+A, 3, 1, 2) = ")


    B = rep(A, 3, 2, 1)
    call display(B, "rep(A, 3, 2, 1) = ")

    C = rep(A+A, 3, 2, 1)
    call display(C, "rep(A+A, 3, 2, 1) = ")


    B = rep(A, 3, 2, 2)
    call display(B, "rep(A, 3, 2, 2) = ")

    C = rep(A+A, 3, 2, 2)
    call display(C, "rep(A+A, 3, 2, 2) = ")


    B = rep(A, 3, 3, 1)
    call display(B, "rep(A, 3, 3, 1) = ")

    C = rep(A+A, 3, 3, 1)
    call display(C, "rep(A+A, 3, 3, 1) = ")


    B = rep(A, 3, 3, 2)
    call display(B, "rep(A, 3, 3, 2) = ")

    C = rep(A+A, 3, 3, 2)
    call display(C, "rep(A+A, 3, 3, 2) = ")

  end subroutine

  subroutine test_sum()
    implicit none
    type(array) :: A, B, C, D

    A = ones(m, n, k, dt = OA_INT)
    B = seqs(m, n, k, dt = OA_INT)

    call display(A, "A = ")
    call display(B, "B = ")

    C = csum(A, 1)
    call display(C, "csum(A, 1) = ")

    C = csum(B, 1)
    call display(C, "csum(B, 1) = ")

    C = csum(A+B, 1)
    call display(C, "csum(A+B, 1) = ")

    D = sum(B, 1)
    call display(D, "sum(B, 1) = ")

    C = csum(A, 2)
    call display(C, "csum(A, 2) = ")

    C = csum(B, 2)
    call display(C, "csum(B, 2) = ")

    C = csum(A+B, 2)
    call display(C, "csum(A+B, 2) = ")

    D = sum(B, 2)
    call display(D, "sum(B, 2) = ")

    C = csum(A, 3)
    call display(C, "csum(A, 3) = ")

    C = csum(B, 3)
    call display(C, "csum(B, 3) = ")

    C = csum(A+B, 3)
    call display(C, "csum(A+B, 3) = ")

    D = sum(B, 3)
    call display(D, "sum(B, 3) = ")

    A = ones(m, n, k, dt = OA_DOUBLE)
    B = seqs(m, n, k, dt = OA_DOUBLE)

    call display(A, "A = ")
    call display(B, "B = ")

    C = csum(A, 1)
    call display(C, "csum(A, 1) = ")

    C = csum(B, 1)
    call display(C, "csum(B, 1) = ")

    C = csum(A+B, 1)
    call display(C, "csum(A+B, 1) = ")

    D = sum(B, 1)
    call display(D, "sum(B, 1) = ")

    C = csum(A, 2)
    call display(C, "csum(A, 2) = ")

    C = csum(B, 2)
    call display(C, "csum(B, 2) = ")

    C = csum(A+B, 2)
    call display(C, "csum(A+B, 2) = ")

    D = sum(B, 2)
    call display(D, "sum(B, 2) = ")

    C = csum(A, 3)
    call display(C, "csum(A, 3) = ")

    C = csum(B, 3)
    call display(C, "csum(B, 3) = ")

    C = csum(A+B, 3)
    call display(C, "csum(A+B, 3) = ")

    D = sum(B, 3)
    call display(D, "sum(B, 3) = ")

    A = ones(m, n, k, dt = OA_FLOAT)
    B = seqs(m, n, k, dt = OA_FLOAT)

    call display(A, "A = ")
    call display(B, "B = ")

    C = csum(A, 1)
    call display(C, "csum(A, 1) = ")

    C = csum(B, 1)
    call display(C, "csum(B, 1) = ")

    C = csum(A+B, 1)
    call display(C, "csum(A+B, 1) = ")

    D = sum(B, 1)
    call display(D, "sum(B, 1) = ")

    C = csum(A, 2)
    call display(C, "csum(A, 2) = ")

    C = csum(B, 2)
    call display(C, "csum(B, 2) = ")

    C = csum(A+B, 2)
    call display(C, "csum(A+B, 2) = ")

    D = sum(B, 2)
    call display(D, "sum(B, 2) = ")

    C = csum(A, 3)
    call display(C, "csum(A, 3) = ")

    C = csum(B, 3)
    call display(C, "csum(B, 3) = ")

    C = csum(A+B, 3)
    call display(C, "csum(A+B, 3) = ")

    D = sum(B, 3)
    call display(D, "sum(B, 3) = ")


  end subroutine test_sum

  subroutine test_min_max()
    implicit none
    type(array) :: A, B, C, D
    real :: v1
    real(kind=8) :: v2
    integer :: v3(3)

    A = rands(4, 3, 2)
    B = rands(4, 3, 2)
    call display(A, "A = ")
    call display(B, "B = ")

    C = min(A, B)
    call display(C, "min(A, B) = ")

    C = max(A, B)
    call display(C, "max(A, B) = ")

    C = max(A*2, B)
    call display(C, "man(A*2, B) = ")

    call set(v1, max(A))

    if(rank == 0) &
      print*, "v1 = ", v1

    call set(v2, min(A))

    if(rank == 0) &
      print*, "v2 = ", v2


    A = rands(4,3,2) - 0.5

    call display(A, "A = ")
    call set(v1, max(A))
    if(rank == 0) then
      print*, "max(A) = ", v1
    end if

    call set(v1, min(A))
    if(rank == 0) then
      print*, "min(A) = ", v1
    end if

    call set(v1, abs_max(A))
    if(rank == 0) then
      print*, "abs_max(A) = ", v1
    end if

    call set(v1, abs_min(A))
    if(rank == 0) then
      print*, "abs_min(A) = ", v1
    end if

    call set(v3, max_at(A))
    if(rank == 0) then
      print*, "max_at(A) = ", v3
    end if

    call set(v3, min_at(A))
    if(rank == 0) then
      print*, "min_at(A) = ", v3
    end if

    call set(v3, abs_max_at(A))
    if(rank == 0) then
      print*, "abs_max_at(A) = ", v3
    end if

    call set(v3, abs_min_at(A))
    if(rank == 0) then
      print*, "abs_min_at(A) = ", v3
    end if

  end subroutine

  subroutine test_operator()
    implicit none
    type(array) :: A, B, C

    !call grid_init('C', dx, dy, dz)

    A = rands(4,4,4) !seqs(4, 4, 4) + 1
    call display(A, "A = ")


    B = AXB(A)
    call display(A, "A = ")
    call display(B, "AXB(A)")


    B = AXF(A)
    call display(A, "A = ")
    call display(B, "AXF(A)")


    B = AYB(A)
    call display(A, "A = ")
    call display(B, "AYB(A)")


    B = AYF(A)
    call display(A, "A = ")
    call display(B, "AYF(A)")


    B = AZB(A)
    call display(A, "A = ")
    call display(B, "AZB(A)")


    B = AZF(A)
    call display(A, "A = ")
    call display(B, "AZF(A)")


    B = DXB(A)
    call display(A, "A = ")
    call display(B, "DXB(A)")


    B = DXF(A)
    call display(A, "A = ")
    call display(B, "DXF(A)")


    B = DYB(A)
    call display(A, "A = ")
    call display(B, "DYB(A)")


    B = DYF(A)
    call display(A, "A = ")
    call display(B, "DYF(A)")


    B = DZB(A)
    call display(A, "A = ")
    call display(B, "DZB(A)")


    B = DZF(A)
    call display(A, "A = ")
    call display(B, "DZF(A)")


    B = DXC(A)
    call display(A, "A = ")
    call display(B, "DXC(A)")


    B = DYC(A)
    call display(A, "A = ")
    call display(B, "DYC(A)")


    B = DZC(A)
    call display(A, "A = ")
    call display(B, "DZC(A)")


    A = rands(m, n, 1)
    B = AXB(A)
    call display(B, "AXB(A)")
    B = AXF(A)
    call display(B, "AXF(A)")
    B = DXB(A)
    call display(B, "DXB(A)")
    B = DXF(A)
    call display(B, "DXF(A)")
    B = DXC(A)
    call display(B, "DXC(A)")

    A = rands(1, n, k)

    B = AZB(A)
    call display(A, "A = ")
    call display(B, "AZB(A)")

    B = AZF(A)
    call display(A, "A = ")
    call display(B, "AZF(A)")

    B = DZB(A)
    call display(A, "A = ")
    call display(B, "DZB(A)")

    B = DZF(A)
    call display(A, "A = ")
    call display(B, "DZF(A)")

    B = DZC(A)
    call display(A, "A = ")
    call display(B, "DZC(A)")

    A = rands(1, 1, k)
    B = AZB(A)
    call display(A, "A = ")
    call display(B, "AZB(A)")
    B = AZF(A)
    call display(A, "A = ")
    call display(B, "AZF(A)")
    B = DZB(A)
    call display(A, "A = ")
    call display(B, "DZB(A)")
    B = DZF(A)
    call display(A, "A = ")
    call display(B, "DZF(A)")
    B = DZC(A)
    call display(A, "A = ")
    call display(B, "DZC(A)")

    A = rands(6, 6, 6)
    call display(A, "A = ")
    B = DXB(AXF(A))
    call display(B, "DXB(AXF(A))")

  end subroutine

  subroutine test_set()
    implicit none
    type(array) :: A, B, C
    integer :: val1
    real  :: val2
    real(kind=8)  :: val3
    integer, allocatable :: farr_int1(:)
    real, allocatable :: farr_float1(:)
    real(8), allocatable :: farr_double1(:)
    integer, allocatable :: farr_int2(:,:)
    real, allocatable :: farr_float2(:,:)
    real(8), allocatable :: farr_double2(:,:)
    integer, allocatable :: farr_int3(:,:,:)
    real, allocatable :: farr_float3(:,:,:)
    real(8), allocatable :: farr_double3(:,:,:)

    call format_short()
    
    val1 = 9
    val2 = 9.900
    val3 = 9.9900



    ! A = seqs(8, 8, 4, dt = OA_INT);
    ! !call set(A, [3,5], [2,5],[1,3], val1)
    ! call set(sub(A,[3,5],[2,5],[1,3]), val1)
    ! !call set_with_const(A, [3,5], [2,5],[1,3], val1)
    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_FLOAT);
    ! !call set(A, [3,5], [2,5],[1,3], val2)
    ! call set(sub(A,[3,5],[2,5],[1,3]), val2)
    ! !call set_with_const(A, [3,5], [2,5],[1,3], val2)
    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_DOUBLE);
    ! !call set(A, [3,5], [2,5],[1,3], val3)
    ! call set(sub(A,[3,5],[2,5],[1,3]), val3)
    ! !call set_with_const(A, [3,5], [2,5],[1,3], val3)
    ! call display(A, "A = ")



    ! A = seqs(8, 8, 4, dt = OA_INT);
    ! B = seqs(3, 4, 3, dt = OA_INT);
    ! !call set(A, [3,5], [2,5],[1,3], B)
    ! call set(sub(A, [3,5], [2,5],[1,3]), B)

    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_FLOAT);
    ! B = seqs(3, 4, 3, dt = OA_FLOAT);
    ! !call set(A, [3,5], [2,5],[1,3], B)
    ! call set(sub(A, [3,5], [2,5],[1,3]), B)

    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_DOUBLE);
    ! B = seqs(3, 4, 3, dt = OA_DOUBLE);
    ! !call set(A, [3,5], [2,5],[1,3], B)
    ! call set(sub(A, [3,5], [2,5],[1,3]), B)

    ! call display(A, "A = ")



    ! A = seqs(8, 8, 4, dt = OA_INT);
    ! B = ones(8, 8, 4, dt = OA_INT);
    ! call set(sub(A, [3,5], [2,5],[1,3]), &
    !   sub(B, [1,3], [1,4], [1,3]))
    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_FLOAT);
    ! B = ones(8, 8, 4, dt = OA_FLOAT);
    ! call set(sub(A, [3,5], [2,5],[1,3]), &
    !   sub(B, [1,3], [1,4], [1,3]))
    ! call display(A, "A = ")


    ! A = seqs(8, 8, 4, dt = OA_DOUBLE);
    ! B = ones(8, 8, 4, dt = OA_DOUBLE);
    ! call set(sub(A, [3,5], [2,5],[1,3]), &
    !   sub(B, [1,3], [1,4], [1,3]))
    ! call display(A, "A = ")



    A = ones(8, 8, 4)
    A = 1.1;
    call display(A, "A = ")

    call set(sub(A,[3,5],[2,5],[1,3]), 2.1_8);
    call display(A, "A = ")

    
    ! allocate(farr_int1(10))
    ! farr_int1(:) = 10
    ! A = farr_int1
    ! call display(A, "A = ")
    
    
    ! allocate(farr_float1(10))
    ! farr_float1(:) = 10
    ! A = farr_float1
    ! call display(A, "A = ")
    
    
    ! allocate(farr_double1(10))
    ! farr_double1(:) = 10
    ! A = farr_double1
    ! call display(A, "A = ")
    
    
    ! allocate(farr_int2(5,5))
    ! farr_int2(:,:) = 10
    ! A = farr_int2
    ! call display(A, "A = ")
    
    
    ! allocate(farr_float2(5,5))
    ! farr_float2(:,:) = 10
    ! A = farr_float2
    ! call display(A, "A = ")
    
    
    ! allocate(farr_double2(5,5))
    ! farr_double2(:,:) = 10
    ! A = farr_double2
    ! call display(A, "A = ")
    
    
    ! allocate(farr_int3(5,4,3))
    ! farr_int3(:,:,:) = 10
    ! A = farr_int3
    ! call display(A, "A = ")
    
    
    ! allocate(farr_float3(5,4,3))
    ! farr_float3(:,:,:) = 10
    ! A = farr_float3
    ! call display(A, "A = ")
    
    
    ! allocate(farr_double3(5,4,3))
    ! farr_double3(:,:,:) = 10
    ! A = farr_double3
    ! call display(A, "A = ")
    

    
    A = seqs(10,10,10)

    if(allocated(farr_double1)) deallocate(farr_double1)
    allocate(farr_double1(10))
    farr_double1 = 10
    
    call set(sub(A, [1,10],1, 1),  farr_double1)

    A = seqs(10,10,10)

    if(allocated(farr_double2)) deallocate(farr_double2)
    allocate(farr_double2(5, 5))
    farr_double1 = 10
    
    call set(sub(A, [1,5], [1,5],[1,1]),  farr_double2)
    call display(A, "A = ")

    A = seqs(10,10,10)

    if(allocated(farr_double3)) deallocate(farr_double3)
    allocate(farr_double3(5, 4, 3))
    farr_double1 = 10
    
    call set(sub(A, [1,5], [1,4],[1,3]),  farr_double3)
    call display(A, "A = ")

    A = seqs(10,10,10)
    deallocate(farr_double3)
    allocate(farr_double3(1,5,5))
    farr_double3 = 10
    call set(sub(A,1,[1,5],[1,5]), farr_double3)

    call set_ref_farray_double_3d(sub(A,1,[1,5],[1,5]), &
      farr_double3)
    call display(A, "A = ")

    A = seqs(10,10,10)
    deallocate(farr_double3)
    allocate(farr_double3(1,1,5))
    farr_double3 = 10
    call set(sub(A,1,1,[1,5]), farr_double3)
    call display(A, "A = ")
    
    call set(val3, sub(a, 1, 1, 1))
    print*, "val3 = ", val3

    call set(val3, sub(a, 1, 2, 1))
    print*, "val3 = ", val3

    call set(val3, sub(a+a, 3, 2, 2))
    print*, "val3 = ", val3

    if(allocated(farr_double3)) &
         deallocate(farr_double3)
    
    allocate(farr_double3(1,1,10))
    
    farr_double3 = sub(A, 1,2,[1,10])
    if(rank == 0) &
         print*, "farr_double3 = ", farr_double3
      
  end subroutine



  subroutine test_set_with_mask()
    implicit none
    type(array) :: A, B, C
    type(array) :: mask
    type(array) :: temparray
    type(node) :: node_mask
    integer :: val1
    real  :: val2
    real(kind=8)  :: val3


    val1 = 9
    val2 = 9.900
    val3 = 9.9900



    A = ones(m, n, k, dt = OA_INT) * 100;
    B = seqs(m, n, k, dt = OA_INT);
    
    ! call display(B, "B = ")
    
    ! B = seqs(m, n, k, dt = OA_INT);

    ! ! mask = seqs(m, n, k, dt = OA_INT);
    mask = (rands(m, n, k) > rands(m, n, k))
    node_mask = mask + mask

    call display(mask, "mask = ")
    
    ! A = ones(m, n, k, dt = OA_INT) * 30;
    ! call set(A, val1, mask)
    ! call display(A, "A1 = ")

    ! A = ones(m, n, k, dt = OA_INT) * 30;
    ! call set(A, val1, A > val1)
    ! call display(A, "A2 = ")

    ! A = ones(m, n, k, dt = OA_INT) * 30;
    ! call set(A, B, A > B)
    ! call display(A, "A3 = ")

    A = ones(m, n, k, dt = OA_INT) * 30;
    call set(A, B, mask)
    call display(A, "A4 = ")

    ! A = ones(m, n, k, dt = OA_INT) * 30;    
    ! call set(A, B * 2, mask)
    ! call display(A, "A5 = ")

    ! A = ones(m, n, k, dt = OA_INT) * 30;    
    ! call set(A, B * 2, A > B * 2)
    ! call display(A, "A6 = ")


    A = ones(m, n, k, dt = OA_FLOAT) * 100;
    B = seqs(m, n, k, dt = OA_FLOAT);
    
    ! call display(B, "B = ")
    
    ! B = seqs(m, n, k, dt = OA_FLOAT);

    ! ! mask = seqs(m, n, k, dt = OA_FLOAT);
    mask = (rands(m, n, k) > rands(m, n, k))
    node_mask = mask + mask

    call display(mask, "mask = ")
    
    ! A = ones(m, n, k, dt = OA_FLOAT) * 30;
    ! call set(A, val2, mask)
    ! call display(A, "A1 = ")

    ! A = ones(m, n, k, dt = OA_FLOAT) * 30;
    ! call set(A, val2, A > val1)
    ! call display(A, "A2 = ")

    ! A = ones(m, n, k, dt = OA_FLOAT) * 30;
    ! call set(A, B, A > B)
    ! call display(A, "A3 = ")

    A = ones(m, n, k, dt = OA_FLOAT) * 30;
    call set(A, B, mask)
    call display(A, "A4 = ")

    ! A = ones(m, n, k, dt = OA_FLOAT) * 30;    
    ! call set(A, B * 2, mask)
    ! call display(A, "A5 = ")

    ! A = ones(m, n, k, dt = OA_FLOAT) * 30;    
    ! call set(A, B * 2, A > B * 2)
    ! call display(A, "A6 = ")


    A = ones(m, n, k, dt = OA_DOUBLE) * 100;
    B = seqs(m, n, k, dt = OA_DOUBLE);
    
    ! call display(B, "B = ")
    
    ! B = seqs(m, n, k, dt = OA_DOUBLE);

    ! ! mask = seqs(m, n, k, dt = OA_DOUBLE);
    mask = (rands(m, n, k) > rands(m, n, k))
    node_mask = mask + mask

    call display(mask, "mask = ")
    
    ! A = ones(m, n, k, dt = OA_DOUBLE) * 30;
    ! call set(A, val3, mask)
    ! call display(A, "A1 = ")

    ! A = ones(m, n, k, dt = OA_DOUBLE) * 30;
    ! call set(A, val3, A > val1)
    ! call display(A, "A2 = ")

    ! A = ones(m, n, k, dt = OA_DOUBLE) * 30;
    ! call set(A, B, A > B)
    ! call display(A, "A3 = ")

    A = ones(m, n, k, dt = OA_DOUBLE) * 30;
    call set(A, B, mask)
    call display(A, "A4 = ")

    ! A = ones(m, n, k, dt = OA_DOUBLE) * 30;    
    ! call set(A, B * 2, mask)
    ! call display(A, "A5 = ")

    ! A = ones(m, n, k, dt = OA_DOUBLE) * 30;    
    ! call set(A, B * 2, A > B * 2)
    ! call display(A, "A6 = ")



  end subroutine

  subroutine test_partition()
    integer :: A(3), B(3), C(3)

    ! call get_procs_shape(A)
    ! print*, "A = ", A

    ! B = [1,2,3]
    ! call set_procs_shape(B)
    ! call get_procs_shape(C)
    ! print*, "C = ", C

    ! call set_auto_procs_shape()
    ! call get_procs_shape(C)
    ! print*, "C = ", C

  end subroutine test_partition

  subroutine test_cache()
    implicit none

    type(array) :: A, B, C, D, E
    type(node) :: ND
    integer :: i

    A = ones(m, n, k)
    B = seqs(m, n, k)
    C = ones(m, n, k) * 0.1;

    do i = 0, 10
      FSET(D, A + B + C)
      FSET(E, A + B - C)       
    end do

    call display(A, "A = ")
    call display(B, "B = ")
    call display(C, "C = ")

    call display(D, "D = ")
    call display(E, "E = ")

  end subroutine

  subroutine test_grid1()
    implicit none
    type(array) :: dx, dy, dz, dt, res

    dx = load("dx.nc", "data")
    dy = load("dy.nc", "data")
    dz = load("dz.nc", "data")
    dt = load("dt.nc", "data")
    
    call grid_init('C', dx, dy, dz)
    call grid_bind(dt, 3)

    call disp(dx, "dx = ")
    call disp(dy, "dy = ")
    call disp(dz, "dz = ")
    call disp(dt, "dt = ")
    
    res = DXB(dt)
    call disp(res, "res = ")
  end subroutine
  
  subroutine test_grid()
    implicit none
    type(array) :: A, B, C, dx, dy, dz
    type(array) :: D, tmp
    print*, "m,n,k=",m,n,k

    D = zeros(m, n, k)

    dx = sub(D, ':', ':', 1)  + 0.1
    dy = sub(D, ':', ':', 1)  + 0.2
    dz = sub(D,  1,   1, ':') + 0.15

    call display(dx, 'dx = ')
    call display(dy, 'dy = ')
    call display(dz, 'dz = ')

    tmp = rep(dz, 2, 2, 1)
    call display(tmp, 'tmp = ')

    call grid_init('C', dx, dy, dz)

    !A = seqs(m, n, k) + 1
    A = rands(m, n, k)

    call grid_bind(A, 3)

    call display(A, "A = ")
    print*, "======================="
    B = AXB(A)
    call display(B, "AXB(A) = ")

    B = DXB(A)
    call display(B, "DXB(A) = ")

    B = AXB(A) 
    call display(B, "AXB(A) = ")
    B = AXF(A) 
    call display(B, "AXF(A) = ")
    B = AYB(A) 
    call display(B, "AYB(A) = ")
    B = AYF(A) 
    call display(B, "AYF(A) = ")
    B = DXB(A) 
    call display(B, "DXB(A) = ")
    B = DXF(A) 
    call display(B, "DXF(A) = ")
    B = DYB(A) 
    call display(B, "DYB(A) = ")
    B = DYF(A) 
    call display(B, "DYF(A) = ")

  end subroutine

  subroutine test_shift()
    implicit none
    type(array) :: A
    type(array) :: B, C

    A = seqs(6, 8, 3)
    ! call display(A, "A")

    B = shift(A, 1)

    call display(B, "B = ")

    B = shift(A, -1)
    call display(B, "B = ")

    C = circshift(A, 1)
    call display(C, "circshift(A, 1) = ")

    C = circshift(A, 0, 1, 0)
    call display(C, "circshift(A, 0, 1, 0) = ")
    
    C = circshift(A, 0, 0, 1)
    call display(C, "circshift(A, 0, 0, 1) = ")

    C = circshift(A, -1)
    call display(C, "circshift(A, -1) = ")

    C = circshift(A, 0, -1)
    call display(C, "circshift(A, 0, -1) = ")
    
    C = circshift(A, 0, 0, -1)
    call display(C, "circshift(A, 0, 0, -1) = ")

    C = circshift(A, 1, -1, 0)
    call display(C, "circshift(A, 1, -1, 0) = ")

    C = circshift(A, -1, 0, 1)
    call display(C, "circshift(A, -1, 0, 1) = ")
    
    
    ! B = shift(A, 1, 2,0)

    ! print*, "after shifted"
    ! call display(A, "A")
  end subroutine


  subroutine test_io()
    implicit none
    type(array) :: A
    type(array) :: B
    type(array) :: C

    A = seqs(6, 8, 3)
    B = ones(2, 4, 3)
    C = zeros(3, 5, 7)

    call save(A, "test_io.nc", "a")
    call save(B, "test_io.nc", "b")
    call save(C, "test_io.nc", "c")

    ! B = load("A.nc", "data")

    ! call display(B, "B = ")

    ! A = zeros(500, 10000, dt=OA_DOUBLE, sw=2)

    ! call save(A, "A1.nc", "data")

    ! B = load("A1.nc", "data")

    ! call display(B, "B1 = ")
    
  end subroutine

  subroutine test_get_ptr()
    implicit none
    type(array) :: A
    type(array) :: B
    real, pointer :: p(:,:,:)
    integer :: i, j, k, s(3)

    A = ones(4, 3, 2)

    call display(A, "A = ")

    call get_local_buffer(p, A)
    print*, "shape(p) = ", shape(p)

    ! s = shape(p)
    ! do k = 1, s(3)
    !    do j = 1, s(2)
    !       do i = 1, s(1)
    !          print*,p(i, j, k)
    !       end do
    !    end do
    ! end do

    ! print*, "p=", p
  end subroutine

  subroutine test_tic_toc()
    implicit none
    integer :: i
    type(array) :: A, B, C

    do i = 1, 10
      call tic("t1")
      call usleep(10000)
      call toc("t1")
    end do

    call tic("t2")
    A = ones(m, n, k)
    B = seqs(m, n, k)
    C = A + B
    call toc("t2")

    if(get_rank() == 0) then
      call show_timer()
    end if

  end subroutine



  subroutine test_fusion_operator()
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

    ! call test_init(6, 6, 6, MPI_COMM_WORLD)
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

  end subroutine

  subroutine test_internal_q()

    type(array) q2f, q2b, dhb, w, q2
    type(array) u, dt, aam, h, v, dhf
    type(array) q2lf, q2lb, q2l

    type(array) dum, dvm

    real(kind=4) :: dti2

    dti2 = 1.0
    call test_init(6, 6, 6, MPI_COMM_WORLD)

    q2f     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    q2b     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    dhb     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    w       = rands(m, n, k, dt = OA_FLOAT, sw=2)
    q2      = rands(m, n, k, dt = OA_FLOAT, sw=2)
    u       = rands(m, n, k, dt = OA_FLOAT, sw=2)
    dt      = rands(m, n, k, dt = OA_FLOAT, sw=2)
    aam     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    h       = rands(m, n, k, dt = OA_FLOAT, sw=2)
    dum     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    v       = rands(m, n, k, dt = OA_FLOAT, sw=2)
    dvm     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    dhf     = rands(m, n, k, dt = OA_FLOAT, sw=2)
    q2lf    = rands(m, n, k, dt = OA_FLOAT, sw=2)
    q2lb    = rands(m, n, k, dt = OA_FLOAT, sw=2)
    q2l     = rands(m, n, k, dt = OA_FLOAT, sw=2)


    q2f= (q2b*dhb-dti2*(-DZB(AZF(w*q2)) &
      + DXF(AXB(q2)* AZB(u) *AXB(dt) &
      -AZB(AXB(aam))*AXB(h)* dum *DXB(q2b)) &
      +DYF(AYB(q2)*  AZB(v)* AYB(dt) &
      -AZB(AYB(aam))*AYB(h)* dvm *DYB(q2b))))/dhf
    call display(q2f, "nnn------------q2f------------")

    FSET(q2f, (q2b*dhb-dti2*(-DZB(AZF(w*q2))+DXF(AXB(q2)*AZB(u)*AXB(dt)-AZB(AXB(aam))*AXB(h)*dum*DXB(q2b))+DYF(AYB(q2)*AZB(v)*AYB(dt)-AZB(AYB(aam))*AYB(h)*dvm*DYB(q2b))))/dhf)
    call display(q2f, "yyy------------q2f------------")

    q2lf= (q2lb*dhb-dti2*(-DZB(AZF(w*q2l)) &
      +DXF(AXB(q2l)* AZB(u) *AXB(dt) &
      -AZB(AXB(aam))*AXB(h)* dum *DXB(q2lb)) &
      +DYF(AYB(q2l)* AZB(v)* AYB(dt) &
      -AZB(AYB(aam))*AYB(h)* dvm *DYB(q2lb))))/dhf
    call display(q2lf, "nnn------------q2lf------------")

    FSET(q2lf, (q2lb*dhb-dti2*(-DZB(AZF(w*q2l))+DXF(AXB(q2l)*AZB(u)*AXB(dt)-AZB(AXB(aam))*AXB(h)*dum*DXB(q2lb))+DYF(AYB(q2l)*AZB(v)*AYB(dt)-AZB(AYB(aam))*AYB(h)*dvm*DYB(q2lb))))/dhf)
    call display(q2lf, "yyy------------q2lf------------")

    q2f= DZB(AZF(w*q2)) &
      + DXF(AXB(q2)*AXB(dt)*AZB(u) &
      - AZB(AXB(aam))*AXB(h)*DXB(q2b)*dum) &
      + DYF(AYB(q2)*AYB(dt)*AZB(v) &
      - AZB(AYB(aam))*AYB(h)*DYB(q2b)*dvm)
    call display(q2f, "nnn------------q2f------------")

    FSET(q2f, DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt)*AZB(u)-AZB(AXB(aam))*AXB(h)*DXB(q2b)*dum)+DYF(AYB(q2)*AYB(dt)*AZB(v)-AZB(AYB(aam))*AYB(h)*DYB(q2b)*dvm))
    call display(q2f, "yyy------------q2f------------")

  end subroutine

  subroutine test_pseudo()
    integer ierr, num_procs, my_id
    CHARACTER(len=32) :: arg
    integer di,dj,dk
    integer step
    integer s
    type(array) :: A, B, C
    real *8 :: start, finish


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

    type(array) :: drhox, dt, mat_ones
    real :: ramp
    integer :: t1,t2
    real time
    call test_init(6, 6, 6, MPI_COMM_WORLD)

    step = 10
   
m=512
n=256
k=48 
    q2f1    = rands(m, n, k, dt = OA_FLOAT, sw=1)
    q2f     = rands(m, n, k, dt = OA_FLOAT, sw=1)
    w       = rands(m, n, k, dt = OA_FLOAT, sw=1)
    q2      = rands(m, n, k, dt = OA_FLOAT, sw=1)
    dt_3d   = rands(m, n, 1, dt = OA_FLOAT, sw=1)
    u       = rands(m, n, k, dt = OA_FLOAT, sw=1)
    aam     = rands(m, n, k, dt = OA_FLOAT, sw=1)
    h_3d    = rands(m, n, 1, dt = OA_FLOAT, sw=1)
    q2b     = rands(m, n, k, dt = OA_FLOAT, sw=1)
    dum_3d  = rands(m, n, 1, dt = OA_FLOAT, sw=1)
    v       = rands(m, n, k, dt = OA_FLOAT, sw=1)
    dvm_3d  = rands(m, n, 1, dt = OA_FLOAT, sw=1)

    !call display(dt_3d, "==============dt_3d===============")
    !FSET(q2f,a1+a2*u)

      FSET(q2f,DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d))

    

    ! ramp = 0.1
 
    ! drhox   = rands(m, n, k, dt = OA_FLOAT, sw=1)
    ! dt      = rands(m, n, 1, dt = OA_FLOAT, sw=1)
    ! mat_ones = ones(m, n, k, dt = OA_FLOAT, sw=1)
 
    ! do s = 1, 3
    !   drhox=(AXB(dt)*mat_ones)**ramp
    !   ! FSET(drhox, ramp*AXB(dt)*mat_ones)
    !   call display(drhox, "==============drhox===============")
    ! end do
    ! 
    ! 
    ! !call display(drhox, "==============drhox===============")
    

    call system_clock(t1)
    start = omp_get_wtime()

    do s = 1 , step
      !q2f1=DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d)
      FSET(q2f,DZB(AZF(w*q2))+DXF(AXB(q2)*AXB(dt_3d)*AZB(U)-AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)+DYF(AYB(q2)*AYB(dt_3d)*AZB(v)-AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d))
    end do

    call system_clock(t2)
    finish = omp_get_wtime()

    !call display(q2f1, "==============q2f1===============")
    !call display(q2f, "==============q2f===============")
if(rank .eq. 0) print '(" ",f6.3,"")',finish-start
if(rank .eq. 0) write(*,*) "time is", t2-t1
  end subroutine

  subroutine test_simple_stmt

  implicit none

  integer ierr, num_procs, my_id
  CHARACTER(len=32) :: arg
  integer di,dj,dk
  integer step
  integer s
  type(array) :: A, B, C
  real :: start, finish

  type(array) ::  a1
  type(array) ::  a2
  type(array) ::  a3
  type(array) ::  a4
  type(array) ::  a5
  type(array) ::  a6
  type(array) ::  a7
  type(array) ::  a8
  type(array) ::  a9
  type(array) :: a10
  type(array) :: a11
  type(array) :: a12
  type(array) :: a13
  type(array) :: a14
  type(array) :: a15
  type(array) :: res
  real :: temp
  real :: X1=1,X2=1,X3=1,X4=1,X5=1,X6=1,X7=1,X8=1,X9=1,X10=1
  real :: Y1=1,Y2=1,Y3=1,Y4=1,Y5=1,Y6=1,Y7=1,Y8=1,Y9=1,Y10=1
  integer :: N1=2,N2=2,N3=2
  integer :: i_error

  step = 10 


!  call test_init(di, dj, dk, MPI_COMM_WORLD) 
m=512
n=256
k=48
   a1 = seqs(m, n, k, dt = OA_FLOAT)
   a2 = seqs(m, n, k, dt = OA_FLOAT)
   a3 = seqs(m, n, k, dt = OA_FLOAT)
   a4 = seqs(m, n, k, dt = OA_FLOAT)
   a5 = seqs(m, n, k, dt = OA_FLOAT)
   a6 = seqs(m, n, k, dt = OA_FLOAT)
   a7 = seqs(m, n, k, dt = OA_FLOAT)
   a8 = seqs(m, n, k, dt = OA_FLOAT)
   a9 = seqs(m, n, k, dt = OA_FLOAT)
  a10 = seqs(m, n, k, dt = OA_FLOAT)
  a11 = seqs(m, n, k, dt = OA_FLOAT)
  a12 = seqs(m, n, k, dt = OA_FLOAT)
  a13 = seqs(m, n, k, dt = OA_FLOAT)
  a14 = seqs(m, n, k, dt = OA_FLOAT)
  a15 = seqs(m, n, k, dt = OA_FLOAT)
  res = seqs(m, n, k, dt = OA_FLOAT)
  
    FSET(res, X1*(-((a1)*(a2))))
    FSET(res, ((((X1*a1+Y1)-(X2*a2))+(X3*a3))-(X4*a4))+((X5*a5)*(a6)))
    FSET(res, (((X1*a1+Y1)+(X2*a2))-(X3*a3))+(X4*a4+Y2))
    FSET(res, (a1)+(((X1*a2)/((a3)*(a4)))*(X3*((X2*a5)/((a6)*(a7)))+Y1)))
    FSET(res, (a1)*(sqrt(a2)))
    FSET(res, ((X1*a1)*(a2))*(a3))
    FSET(res, (-(((a1)*(a2))*(a3)))+((((a4)*(a5))*(a6))*(a7)))
    FSET(res, (X1*a1)+((a2)*(a3)))
    FSET(res, ((a1)*(a2))+(X1*a3))
    FSET(res, -((a1)*(a2+Y1)))
    FSET(res, (((a1)*(a2))-((a3)*(a4)))/((a5)*(a6)))
    FSET(res, ((a1)+(a2))-(a3))
    FSET(res, ((a1)*(a2))-((X1*((a3)*(a4)))*(a5)))
    FSET(res, ((a1)*(a2))-(((a3)*(a4))*((a5)+(a6))))
    FSET(res, ((a1)*(a2))*(a3))
    FSET(res, ((a1)+(a2))+(a3))
    FSET(res, ((a1)*(a2))*((a3)+(a4)))
    FSET(res, ((a1)*(a2))-(((X1*a3)*(a4))*(a5)))
    FSET(res, ((a1)*(a2))-(a3))
    FSET(res, (a1)-(X1*(((a2)+(a3))-(a4))))
    FSET(res, (((a1)*(a2))-(X4*((((((a3)+(a4))-(a5))+((X1*a6)*(((X2*a7)+(X3*((a8)+(a9))))+(a10))))+(a11))+((a12)-(a13)))))/(a14))
    FSET(res, X3*((a1)+((X2*(sqrt(X1*(1.0/a2))))*((a3)-(a4)))))
    FSET(res, X3*((a1)-((X2*(sqrt(X1*(1.0/a2))))*((a3)-(a4)))))
    FSET(res, (((a1)*(a2))-(X4*((((((a3)+(a4))+(a5))+((X1*a6)*(((X2*a7)+(X3*((a8)+(a9))))+(a10))))+(a11))+((a12)-(a13)))))/(a14))
    FSET(res, (a1)+(X2*(((a2)-(X1*a3))+(a4))))
    FSET(res, (a1)+(X2*((X1*a2)*(a3))))
    FSET(res, ((a1)+(X1*a2))*(a3))
    FSET(res, ((a1)-(a2))+((a3)/(X1*a4)))
    FSET(res, (a1)*(((a2)+(a3))+(a4)))
    FSET(res, ((a1)+(a2))*(a3))
    FSET(res, (((a1)*((a2)+(a3)))-(X1*(((-(a4))+(a5))+(a6))))/((a7)+(a8)))
    FSET(res, (((a1)*(a2))*(a3))-((((a4)*(a5))*(a6))*(a7)))
    FSET(res, -((X1*a1+Y1)/((((a2)*(a3))*(a4))*(a5))))
    FSET(res, -((X1*a1+Y1)/(((a2)*(a3))*(a4))))
    FSET(res, (-((X1*a1)/(a2)))+(X2*(1.0/a3)))
    FSET(res, ((a1)*(a2))+((a3)*(a4)))
    FSET(res, (sqrt(a1))/(X1*a2+Y1))
    FSET(res, (X1*a1+Y1)/(X3*((X2*(1.0/a2)+Y2)*(a3))+Y3))
    FSET(res, ((X1*a1+Y1)+((X2*a2)*(a3)))/(X3*a4+Y2))
    FSET(res, (a1)*(sqrt(abs(a2))))
    FSET(res, X2*(((X1*a1)*(a2))+(a3)))
    FSET(res, X1*(((a1)*(a2))+(a3)))
    FSET(res, (a1)+(X2*(((a2)+(a3))-(X1*a4))))
    FSET(res, ((((a1)+(a2))*(a3))-(X1*(((a4)+(a5))-(a6))))/((a7)+(a8)))
    FSET(res, (((a1)*(a2))*(a3))-(((X1*((a4)*(a5)))*(a6))*(a7)))
    FSET(res, (a1)/((((a2)*(a3))*(a4))*(a5)))
    FSET(res, ((-((X1*a1)/(-(X2*a2))))-(a3))/(a4+Y1))
    FSET(res, (a1)-(X3*((((X1*((a2)-(abs(a3))))*((a4)-(a5)))/(a6))+((X2*((a7)+(abs(a8))))*(a9)))))
    FSET(res, (a1)+(((((((X1*((a2)+(abs(a3))))/(a4))*(a5))/(a6))*(a7))*(a8))/(a9)))
    FSET(res, (a1)-(X3*((((X1*((a2)+(abs(a3))))*((a4)-(a5)))/(a6))+((((X2*((a7)-(abs(a8))))*(a9))*(a10))/(a11)))))
    FSET(res, (a1)+(((((((X1*((a2)-(abs(a3))))/(a4))*(a5))/(a6))*(a7))*(a8))/(a9)))
    FSET(res, (((a1)*(a2))-(X4*(((((a3)+(a4))-(a5))+(X3*((X1*a6)*((a7)+(X2*a8)))))-(a9))))/(a10))
    FSET(res, (((X1*a1)/((a2)*(a3)))-(a4))/(a5+Y1))
    FSET(res, -((a1)*(a2)))
    FSET(res, ((sqrt(X1*a1))*(a2))+((X3*(sqrt(X2*a3))+Y1)*(a4)))
    FSET(res, (((a1)*(a2))-(X4*(((((a3)+(a4))+(a5))+(X3*((X1*a6)*((a7)+(X2*a8)))))-(a9))))/(a10))
    FSET(res, (a1)+(X2*((((a2)+(a3))-(X1*a4))-(a5))))
    FSET(res, (((a1)+(a2))-(X1*a3))*(a4))

  !call MPI_Barrier(MPI_COMM_WORLD,i_error)

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, X1*(-((a1)*(a2))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("1 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((((X1*a1+Y1)-(X2*a2))+(X3*a3))-(X4*a4))+((X5*a5)*(a6)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("2 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((X1*a1+Y1)+(X2*a2))-(X3*a3))+(X4*a4+Y2))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("4 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(((X1*a2)/((a3)*(a4)))*(X3*((X2*a5)/((a6)*(a7)))+Y1)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("5 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)*(sqrt(a2)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("6 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((X1*a1)*(a2))*(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("7 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (-(((a1)*(a2))*(a3)))+((((a4)*(a5))*(a6))*(a7)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("8 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (X1*a1)+((a2)*(a3)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("10 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))+(X1*a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("11 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, -((a1)*(a2+Y1)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("12 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))-((a3)*(a4)))/((a5)*(a6)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("13 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)+(a2))-(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("14 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))-((X1*((a3)*(a4)))*(a5)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("15 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))-(((a3)*(a4))*((a5)+(a6))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("16 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))*(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("17 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)+(a2))+(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("18 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))*((a3)+(a4)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("20 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))-(((X1*a3)*(a4))*(a5)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("21 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))-(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("22 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)-(X1*(((a2)+(a3))-(a4))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("23 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))-(X4*((((((a3)+(a4))-(a5))+((X1*a6)*(((X2*a7)+(X3*((a8)+(a9))))+(a10))))+(a11))+((a12)-(a13)))))/(a14))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("24 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, X3*((a1)+((X2*(sqrt(X1*(1.0/a2))))*((a3)-(a4)))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("25 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, X3*((a1)-((X2*(sqrt(X1*(1.0/a2))))*((a3)-(a4)))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("26 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))-(X4*((((((a3)+(a4))+(a5))+((X1*a6)*(((X2*a7)+(X3*((a8)+(a9))))+(a10))))+(a11))+((a12)-(a13)))))/(a14))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("27 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(X2*(((a2)-(X1*a3))+(a4))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("28 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(X2*((X1*a2)*(a3))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("29 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)+(X1*a2))*(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("30 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)-(a2))+((a3)/(X1*a4)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("31 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)*(((a2)+(a3))+(a4)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("32 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)+(a2))*(a3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("33 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*((a2)+(a3)))-(X1*(((-(a4))+(a5))+(a6))))/((a7)+(a8)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("34 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))*(a3))-((((a4)*(a5))*(a6))*(a7)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("35 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, -((X1*a1+Y1)/((((a2)*(a3))*(a4))*(a5))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("36 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, -((X1*a1+Y1)/(((a2)*(a3))*(a4))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("37 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (-((X1*a1)/(a2)))+(X2*(1.0/a3)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("41 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((a1)*(a2))+((a3)*(a4)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("42 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (sqrt(a1))/(X1*a2+Y1))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("45 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (X1*a1+Y1)/(X3*((X2*(1.0/a2)+Y2)*(a3))+Y3))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("46 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((X1*a1+Y1)+((X2*a2)*(a3)))/(X3*a4+Y2))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("47 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)*(sqrt(abs(a2))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("48 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, X2*(((X1*a1)*(a2))+(a3)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("49 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, X1*(((a1)*(a2))+(a3)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("50 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(X2*(((a2)+(a3))-(X1*a4))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("51 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((((a1)+(a2))*(a3))-(X1*(((a4)+(a5))-(a6))))/((a7)+(a8)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("52 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))*(a3))-(((X1*((a4)*(a5)))*(a6))*(a7)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("53 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)/((((a2)*(a3))*(a4))*(a5)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("54 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((-((X1*a1)/(-(X2*a2))))-(a3))/(a4+Y1))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("55 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)-(X3*((((X1*((a2)-(abs(a3))))*((a4)-(a5)))/(a6))+((X2*((a7)+(abs(a8))))*(a9)))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("56 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(((((((X1*((a2)+(abs(a3))))/(a4))*(a5))/(a6))*(a7))*(a8))/(a9)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("57 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)-(X3*((((X1*((a2)+(abs(a3))))*((a4)-(a5)))/(a6))+((((X2*((a7)-(abs(a8))))*(a9))*(a10))/(a11)))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("58 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(((((((X1*((a2)-(abs(a3))))/(a4))*(a5))/(a6))*(a7))*(a8))/(a9)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("59 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))-(X4*(((((a3)+(a4))-(a5))+(X3*((X1*a6)*((a7)+(X2*a8)))))-(a9))))/(a10))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("60 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((X1*a1)/((a2)*(a3)))-(a4))/(a5+Y1))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("61 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, -((a1)*(a2)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("63 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, ((sqrt(X1*a1))*(a2))+((X3*(sqrt(X2*a3))+Y1)*(a4)))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("64 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)*(a2))-(X4*(((((a3)+(a4))+(a5))+(X3*((X1*a6)*((a7)+(X2*a8)))))-(a9))))/(a10))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("65 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (a1)+(X2*((((a2)+(a3))-(X1*a4))-(a5))))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("66 Time(seconds) = ",f6.3,"")',finish-start

  !call cpu_time(start)
  !do s = 1 , step
  !  FSET(res, (((a1)+(a2))-(X1*a3))*(a4))
  !  call MPI_Barrier(MPI_COMM_WORLD,i_error)
  !end do
  !call cpu_time(finish)
  !print '(" ",f6.3,"")',finish-start
  !!print '("67 Time(seconds) = ",f6.3,"")',finish-start

print * , "~~~~~~~~~~~~~"


!  call oa_finalize()

  end subroutine


  subroutine test_wave()
    implicit none
    type(array) :: u0, ut, res, tmp, u1, u2, x
    integer :: i, steps, nx
    double precision :: dt, dx

    dt = 0.01
    dx = 0.01
    nx = 500
    steps = 10000
    x = seqs(nx) * dx;
    u0 = cos(x*10) * exp(-(x-2.5)**4);
    u1 = shift(u0, 1)

    res = zeros(nx, steps, 1, dt=OA_DOUBLE)
    
    do i = 1, 10000
       print*, "i = ", i
       
       u2 = 2 * u1 - u0 + DXB(DXF(u1)) / (dx*dx) * (dt*dt)
       
       call set(sub(u2, 1),  0)
       call set(sub(u2, nx), 0)
       call set(sub(res, ':', i, 1), u2)
       
       u0 = u1
       u1 = u2
    end do
    call save(res, "result.nc", "data")
  end subroutine 


subroutine wave(nt , nx, ny)
     implicit none
     type(array) :: u_new, u, u_old
     integer,intent(in) :: nx, ny, nt
     real :: sigma, gamma
     sigma = 0.1
     gamma = 0.1

     u = seqs(nx, ny, 1 , dt=OA_DOUBLE)
     u_new = seqs(nx, ny, 1, dt=OA_DOUBLE)
     u_old = seqs(nx, ny, 1, dt=OA_DOUBLE)
     do n=1,nt
        u_new = 2 * ( sigma**2*(DXF(u)-DXB(u)) + gamma**2*(DYF(u)-DYB(u)) ) + 2*u - u_old
        u_old = u
        u = u_new
     enddo
end subroutine

subroutine Runge_Kutte(nt, nx, ny)
  implicit none
  type(array) :: T, Tk, k1, k2
  double precision :: dx, dy,dt, alpha
  integer , intent(in):: nt, nx,ny
  integer :: k
  T = seqs(nx, ny, 1, dt=OA_DOUBLE)
  Tk = seqs(nx, ny, 1, dt=OA_DOUBLE)
  k1 = seqs(nx, ny, 1, dt=OA_DOUBLE)
  k2 = seqs(nx, ny, 1, dt=OA_DOUBLE)
   do k =1,nt
     k1=2*alpha* ( (DXF(T)-DXB(T))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
     Tk=T+k1*dt
     k2=2*alpha* ( (DXF(Tk)-DXB(Tk))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
     T=T+dt/2*(k1+k2)
   enddo
 end subroutine



subroutine Euler(nt,nx,ny)
    implicit none
    type(array):: T
    double precision ::  dt, dx, dy, alpha 
    integer, intent(in):: nx,ny,nt
    integer :: k
    dx = 0.1
    dy = 0.1
    dt = 0.1
    alpha = 0.1

    T = seqs(nx,ny,1, dt=OA_DOUBLE)
    do  k=1,nt
        T = T + dt*alpha*2* ( (DXF(T)-DXB(T))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
    enddo
end subroutine

subroutine height(nt, nx, ny)
  implicit none
  type(array) :: D, U, V ,E
  double precision :: dt
  integer,intent(in) :: nx,ny,nt
  integer :: i
  dt = 0.1
  D = seqs(nx,ny,1, dt=OA_DOUBLE)
  U = seqs(nx,ny,1,dt=OA_DOUBLE)
  V = seqs(nx,ny,1,dt=OA_DOUBLE)
  E = seqs(nx,ny,1,dt=OA_DOUBLE)
  do i=1,nt
    E=E-dt*(DXF(AXB(D)*U)+DYF(AYB(D)*V))
  enddo
end subroutine

subroutine test_plus(nt, nx, ny)
  implicit none
  type(array) :: D, U, V ,E
  integer,intent(in) :: nx,ny,nt
  integer :: i
  D = seqs(nx,ny,1, dt=OA_DOUBLE)
  U = seqs(nx,ny,1,dt=OA_DOUBLE)
  V = seqs(nx,ny,1,dt=OA_DOUBLE)
  E = seqs(nx,ny,1,dt=OA_DOUBLE)
  do i=1,nt
    E=D+1
  enddo
call disp(E,"E=")
end subroutine

subroutine test_slice(nt, nx,ny)
  implicit none
  type(array) :: D, U, V ,E
  integer,intent(in) :: nx,ny,nt
  integer :: x,i
 x =1 
  D = seqs(nx,ny,1, dt=OA_DOUBLE)
  do i=1,nt
  U = slice(D, x)
  enddo
call disp(U,"U=")
end subroutine



  subroutine test_interpolation()
    implicit none
    type(array) :: A, B, C
    type(node) :: nn
    integer :: m,n,k
    
    m = 5
    n = 5
    k = 5

    A = seqs(m, n, k)

    call display(A, "A = ")

!    B = interpolation(A, 2, 2, 2)
!    call display(B, "interpolation(A, 2, 2, 2) = ")
  end subroutine

  subroutine test_mat_mult()
    implicit none
    type(array) :: A,B,C
    type(node) :: nn
    integer :: m, n, k

    !----------------------------------------------------------------
    ! test float

    ! test x and y don't match
    ! A = ones(5, 4, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(5, 4, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "x and y don't match = ")
  
    ! test z don't match
    ! A = ones(5, 4, 2,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 3,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "z don't match = ")

    ! ! test 2d*2d case 1
    ! A = ones(5, 4, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 1 = ")

    ! ! test 2d*2d case 2
    ! A = ones(5, 5, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(5, 5, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 2 = ")

    ! ! test 2d*2d case 3
    ! A = ones(1, 4, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 3 = ")

    ! ! test 2d*2d case 4
    ! A = ones(4, 5, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 4 = ")

    ! ! test 2d*2d case 5
    ! A = ones(1, 5, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 5 = ")

    ! ! test 2d*2d case 6
    ! A = ones(5, 1, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(1, 6, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 6 = ")

    ! ! test 2d*2d case 7
    ! A = ones(17, 7, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(7, 17, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 7 = ")

    ! ! test 3d*3d case 1
    ! A = seqs(2, 3, 2,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 2,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 1 = ")

    ! ! test 3d*3d case 2
    ! A = seqs(2, 3, 5,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 5,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 2 = ")

    ! ! test 2d*3d case 1
    ! A = ones(2, 3, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 3,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 1 = ")

    ! ! test 2d*3d case 2
    ! A = seqs(2, 3, 1,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 3,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 2 = ")

    ! ! test 3d*2d
    ! A = ones(2, 3, 3,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 1 = ")

    ! ! test 3d*2d
    ! A = seqs(2, 3, 3,dt=OA_FLOAT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 1,dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 2 = ")

    ! !----------------------------------------------------------------
    ! ! test int

    ! ! test x and y don't match
    ! ! A = ones(5, 4, 1,dt=OA_INT)
    ! ! call display(A, "A = ")
    ! ! B = seqs(5, 4, 1,dt=OA_INT)
    ! ! call display(B, "B = ")
    ! ! C = mat_mult(A, B)
    ! ! call display(C, "x and y don't match = ")
  
    ! ! test z don't match
    ! ! A = ones(5, 4, 2,dt=OA_INT)
    ! ! call display(A, "A = ")
    ! ! B = seqs(4, 5, 3,dt=OA_INT)
    ! ! call display(B, "B = ")
    ! ! C = mat_mult(A, B)
    ! ! call display(C, "z don't match = ")

    ! ! test 2d*2d case 1
    ! A = ones(5, 4, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 1 = ")

    ! ! test 2d*2d case 2
    ! A = ones(5, 5, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(5, 5, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 2 = ")

    ! ! test 2d*2d case 3
    ! A = ones(1, 4, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 3 = ")

    ! ! test 2d*2d case 4
    ! A = ones(4, 5, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 4 = ")

    ! ! test 2d*2d case 5
    ! A = ones(1, 5, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 5 = ")

    ! ! test 2d*2d case 6
    ! A = ones(5, 1, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(1, 6, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 6 = ")

    ! ! test 2d*2d case 7
    ! A = ones(17, 7, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(7, 17, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 7 = ")

    ! ! test 3d*3d case 1
    ! A = seqs(2, 3, 2,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 2,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 1 = ")

    ! ! test 3d*3d case 2
    ! A = seqs(2, 3, 5,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 5,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 2 = ")

    ! ! test 2d*3d case 1
    ! A = ones(2, 3, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 3,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 1 = ")

    ! ! test 2d*3d case 2
    ! A = seqs(2, 3, 1,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 3,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 2 = ")

    ! ! test 3d*2d
    ! A = ones(2, 3, 3,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 1 = ")

    ! ! test 3d*2d
    ! A = seqs(2, 3, 3,dt=OA_INT)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 1,dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 2 = ")

    ! !----------------------------------------------------------------
    ! ! test double

    ! ! test x and y don't match
    ! ! A = ones(5, 4, 1,dt=OA_DOUBLE)
    ! ! call display(A, "A = ")
    ! ! B = seqs(5, 4, 1,dt=OA_DOUBLE)
    ! ! call display(B, "B = ")
    ! ! C = mat_mult(A, B)
    ! ! call display(C, "x and y don't match = ")
  
    ! ! test z don't match
    ! ! A = ones(5, 4, 2,dt=OA_DOUBLE)
    ! ! call display(A, "A = ")
    ! ! B = seqs(4, 5, 3,dt=OA_DOUBLE)
    ! ! call display(B, "B = ")
    ! ! C = mat_mult(A, B)
    ! ! call display(C, "z don't match = ")

    ! ! test 2d*2d case 1
    ! A = ones(5, 4, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 1 = ")

    ! ! test 2d*2d case 2
    ! A = ones(5, 5, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(5, 5, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 2 = ")

    ! ! test 2d*2d case 3
    ! A = ones(1, 4, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(4, 5, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 3 = ")

    ! ! test 2d*2d case 4
    ! A = ones(4, 5, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 4 = ")

    ! ! test 2d*2d case 5
    ! A = ones(1, 5, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(5, 1, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 5 = ")

    ! ! test 2d*2d case 6
    ! A = ones(5, 1, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(1, 6, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 6 = ")

    ! ! test 2d*2d case 7
    ! A = ones(17, 7, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(7, 17, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*2d case 7 = ")

    ! ! test 3d*3d case 1
    ! A = seqs(2, 3, 2,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 2,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 1 = ")

    ! ! test 3d*3d case 2
    ! A = seqs(2, 3, 5,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 5,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*3d case 2 = ")

    ! ! test 2d*3d case 1
    ! A = ones(2, 3, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 3,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 1 = ")

    ! ! test 2d*3d case 2
    ! A = seqs(2, 3, 1,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 3,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "2d*3d case 2 = ")

    ! ! test 3d*2d
    ! A = ones(2, 3, 3,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = seqs(3, 2, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 1 = ")

    ! ! test 3d*2d
    ! A = seqs(2, 3, 3,dt=OA_DOUBLE)
    ! call display(A, "A = ")
    ! B = ones(3, 2, 1,dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "3d*2d case 2 = ")

    
 !   A = ones(3, 3, 1,dt=OA_FLOAT)
 !   call display(A, "A = ")
 !   B = seqs(3, 3, 1,dt=OA_FLOAT)
 !   call display(B, "B = ")
 !   ! test node * node
 !   C = mat_mult(A+B, A-B)
 !   call display(C, "node * node = ")
 !   ! test node * array
 !   C = mat_mult(A+B, A)
 !   call display(C, "node * array = ")
 !   ! test array * node
 !   C = mat_mult(A, A+B)
 !   call display(C, "array * node = ")
 !   ! test complex 1
 !   C = mat_mult(mat_mult(A, A+B), A)
 !   call display(C, "complex 1 = ")
 !   ! test complex 2
 !   C = mat_mult(A, mat_mult(A, A+B))
 !   call display(C, "complex 2 = ")
 !   ! test complex 3
 !   C = mat_mult(mat_mult(A, A+B), mat_mult(A+B, A))
 !   call display(C, "complex 3 = ")

    ! ! test double * float
    ! A = ones(3, 3, 1, dt=OA_DOUBLE)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "double * float = ")

    ! ! test float * double
    ! A = ones(3, 3, 1, dt=OA_FLOAT)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "float * double = ")

    ! ! test double * int
    ! A = ones(3, 3, 1, dt=OA_DOUBLE)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "double * int = ")

    ! ! test int * double
    ! A = ones(3, 3, 1, dt=OA_INT)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_DOUBLE)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "int * double = ")

    ! ! test float * int
    ! A = ones(3, 3, 1, dt=OA_FLOAT)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "float * int = ")

    ! ! test int * float
    ! A = ones(3, 3, 1, dt=OA_INT)
    ! call display(A, "A = ") 
    ! B = seqs(3, 3, 1, dt=OA_FLOAT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "int * float = ")

    ! ! test ps[2]
    ! A = ones(4, 4, 4, dt=OA_INT)
    ! call display(A, "A = ") 
    ! B = seqs(4, 4, 4, dt=OA_INT)
    ! call display(B, "B = ")
    ! C = mat_mult(A, B)
    ! call display(C, "ps[2] = ")


    ! test done

  end subroutine

 subroutine test_ones()
    implicit none
    type(Array) :: A

    A = ones(3, 3, 3)
call display(A,"A=")
 end subroutine 

end module
