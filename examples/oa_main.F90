#include "../src/fortran/config.h"
module oa_test
  use iso_c_binding
  use oa_mod

contains
  
  subroutine array_creation()
    implicit none
    type(array) :: A, B, C, D
    
    A = zeros(2, 2, 2)
    B = ones(2, 2, 2)
    C = seqs(2, 2, 2)
    D = rands(2, 2, 2)
    
    call display(A, "zeros = ")
    call display(B, "ones = ")
    call display(C, "seqs = ")
    call display(D, "rands = ")
  end subroutine

  subroutine arithmetic_operation()
    implicit none
    type(array) :: A, B, C, D, E, ans 

    ! basic arithmetic operators
    A = zeros(2, 2, 1)
    B = ones(2, 2, 1)
    C = seqs(2, 2, 1)
    D = rands(2, 2, 1)

    E = A + B * C 
    call display(E, "E = ")
    E = E / 2.0 
    call display(E, "E = ")
  
    ! comparison operators
    ans = B < C 
    call display(ans, "ones < seqs = ")
    ans = A == C
    call display(ans, "ones == seqs = ")
    
    ! logical operators
    ans = A .or. C
    call display(ans, "zeros .or. seqs = ")
    ans = A .and. C
    call display(ans, "zeros .and. seqs = ")
    
    ! basic math functions
    E = consts_double(2, 2, 1, 2.D0*atan(1.D0))
    ans = sin(E)
    call display(ans, "sin(PI/2) = ")
    ans = exp(C)
    call display(ans, "exp(seqs) = ")
    ans = log(ans)
    call display(ans, "log(exp(seqs)) = ")

  end subroutine

  subroutine array_operation()
    implicit none
    type(array) :: A, B, ans
    real :: mx
    integer :: pos(3)
    real(4), allocatable :: farray(:,:,:)

    ! get a sub slice of array
    A = seqs(4, 4, 2)
    call display(A, "A = ")
    ! sub function index start from 1
    ans = sub(A, [2, 4], [2, 4], [1, 1])
    call display(ans, "sub(A, [2, 4], [2, 4], [1, 1]) = ")
    ans = sub(A, ':', ':', [2, 2])
    call display(ans, "sub(A, ':', ':', [2, 2]) = ")
    ans = sub(A, 1, 2)
    call display(ans, "sub(A, 1, 2) = ")

    ! shift array in a given direction
    A = seqs(2, 2, 2)
    call display(A, "A = ")
    ans = shift(A, 1, 0, 0)
    call display(ans, "shift(A, 1, 0, 0) = ")
    ans = shift(A, 0, 1)
    call display(ans, "shift(A, 0, 1) = ")
    ans = shift(A, 0, -1, 0)
    call display(ans, "shift(A, 0, -1, 0) = ")
    ans = circshift(A, 0, 1)
    call display(ans, "circshift(A, 0, 1) = ")

    ! sum array in a given direction
    A = seqs(2, 2, 2)
    ans = sum(A, 1)
    call display(ans, "sum(A, 1) = ")

    ! cumulative sum
    ans = csum(A, 2)
    call display(ans, "csum(A, 2) = ")

    ! get maximum value or its position
    A = rands(2, 2, 1)
    B = rands(2, 2, 1)
    call display(A, "A = ")
    call display(B, "B = ")
    ans = max(A, B)
    call display(ans, "max(A, B) = ")
    mx = max(A)
    print *, "max(A) = ", mx
    pos = max_at(A)
    print *, "max_at(A) = ", pos

    ! get minimum value or its position
    ans = min(A, B)
    call display(ans, "min(A, B) = ")
    mx = min(A)
    print *, "min(A) = ", mx

    pos = min_at(A)
    print *, "min_at(A) = ", pos

    ! repeat array in a given direction
    A = seqs(2, 2, 1)
    ans = rep(A, 1, 2, 1)
    call display(ans, "rep(A, 1, 2, 1) = ")

    ! array assignment
    call set(sub(A, [1,2], [1,1], [1,1]), -0.5)
    call display(A, "set(A) = ")

    if(allocated(farray)) deallocate(farray)
    allocate(farray(2, 1, 1))
    farray(:,:,:) = 10
    call set(sub(A, [1,2], [1,1], [1,1]), farray)
    call display(A, "set with fortran array = ")

  end subroutine

  subroutine stencil_operation()
    implicit none
    type(array) :: A, B, ans

    A = seqs(2, 2, 2)

    ! averaging stencil operators
    ans = AXB(A)
    call display(ans, "AXB(A) = ")
    ans = AZF(A)
    call display(ans, "AZF(A) = ")

    ! differential stencil operators
    ans = DXB(A)
    call display(ans, "DXB(A) = ")
    ans = DYF(A)
    call display(ans, "DYF(A) = ")
  end subroutine

  subroutine io_operation()
    implicit none
    type(array) :: A, ans

    A = ones(2, 2, 2)

    ! save array into file
    call save(A, "test_io.nc", "a")

    ! load array from file
    ans = load("test_io.nc", "a")
    call display(ans, "load from file")
  end subroutine

  subroutine util_operation()
    implicit none
    type(array) :: A, ans, dx, dy, dz

    A = zeros(3, 3, 3, dt=OA_DOUBLE)
    dx = sub(A, ':', ':', 1)  + 0.1D0
    dy = sub(A, ':', ':', 1)  + 0.2D0
    dz = sub(A,  1,   1, ':') + 0.15D0

    call display(dx, 'dx = ')
    call display(dy, 'dy = ')
    call display(dz, 'dz = ')

    call tic("grid_init")  ! start the timer
    call grid_init('C', dx, dy, dz)  ! init grid C with dx, dy, dz
    call toc("grid_init")  ! end the timer

    A = seqs(3, 3, 3, dt=OA_DOUBLE)
    ans = 1.0 * DXF(A)
    call display(ans, "DXF(A) = ")

    call grid_bind(A, 3)  ! bind A to point 3
    ans = DXF(A)
    call display(ans, "after binding C gird at point 3, DXF(A) = ")
  end subroutine

  subroutine continuity(nt, nx, ny, nz)
    implicit none
    type(array) :: D, U, V ,E
    real*8:: dt
    integer,intent(in) :: nx, ny, nz, nt
    integer :: k

    ! initialize data with random numbers 
    D = rands(nx, ny, nz, dt=OA_DOUBLE)
    U = D
    V = D
    E = D
    dt = 0.1

    call tic("continuity")
    do k=1,nt
      E = E - 2*dt*(DXF(AXB(D)*U)+DYF(AYB(D)*V))
    enddo
    call toc("continuity")
  end subroutine

  subroutine heat_diffusion(nt, nx, ny, nz)
    implicit none
    type(array):: T
    real*8 :: dt, dx, dy, alpha
    integer, intent(in) :: nx, ny, nz, nt
    integer :: k

    dx = 0.1
    dy = 0.1
    dt = 0.1
    alpha = 0.1
    T = rands(nx, ny, nz, dt=OA_DOUBLE)

    call tic("heat_diffusion")
    do k=1,nt
        T = T + dt*alpha*2*( (DXF(T)-DXB(T))/(dx*dx) + (DYF(T)-DYB(T))/(dy*dy) )
    enddo
    call toc("heat_diffusion")
  end subroutine

  subroutine hotspot2D(nt, nx, ny, nz)
    implicit none
    type(array) :: T, P
    double precision :: dx, dy, Cap, Rx, Ry, Rz, max_slope, dt
    double precision :: chip_height, chip_width, t_chip, FACTOR_CHIP, SPEC_HEAT_SI
    double precision :: K_SI, MAX_PD, PRECISIONN, amb_temp
    integer,intent(in) :: nx,ny,nz,nt
    integer :: i

    ! initialize coefficients 
    PRECISIONN = 0.001
    SPEC_HEAT_SI = 1.75e6
    K_SI = 100
    FACTOR_CHIP = 0.5
    MAX_PD = 3.0e6
    chip_height = 0.016
    chip_width = 0.016
    t_chip = 0.0005
    amb_temp = 80.0

    dx = chip_height/nx
    dy = chip_width/ny

    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy
    Rx = dy / (2.0 * K_SI * t_chip * dx)
    Ry = dx / (2.0 * K_SI * t_chip * dy)
    Rz = t_chip / (K_SI * dx * dy)

    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    dt = PRECISIONN / max_slope / 1000.0

    Cap = dt / Cap
    Rx = 1.0 / Rx
    Ry = 1.0 / Ry
    Rz = 1.0 / Rz

    T = consts_double(nx,ny,nz,233.3D0,1)
    P = consts_double(nx,ny,nz,233.3D0,1)

    call tic("hotspot2D")
    do i=1,nt
      T=T+cap*(P+(DXF(T)-DXB(T))*Ry+(DYF(T)-DYB(T))*Rx+(amb_temp-T)*Rz)
    enddo
    call toc("hotspot2D")
  end subroutine

  subroutine hotspot3D(nt, nx, ny, nz)
    implicit none
    type(array) :: T, P
    double precision :: dx, dy, dz, Cap, Rx, Ry, Rz, max_slope, dt
    double precision :: chip_height, chip_width, t_chip, FACTOR_CHIP, SPEC_HEAT_SI
    double precision :: K_SI, MAX_PD, PRECISIONN, amb_temp
    double precision :: cx, cy, cz, cc, stepDivCap
    integer,intent(in) :: nx,ny,nz,nt
    integer :: i

    ! initialize coefficients 
    PRECISIONN = 0.001
    SPEC_HEAT_SI = 1.75e6
    K_SI = 100
    FACTOR_CHIP = 0.5
    MAX_PD = 3.0e6
    chip_height = 0.016
    chip_width = 0.016
    t_chip = 0.0005
    amb_temp = 80.0

    dx = chip_height/nx
    dy = chip_width/ny
    dz = t_chip/nz

    Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy
    Rx = dy / (2.0 * K_SI * t_chip * dx)
    Ry = dx / (2.0 * K_SI * t_chip * dy)
    Rz = dz / (K_SI * dx * dy)

    max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI)
    dt = PRECISIONN / max_slope

    T = consts_double(nx,ny,nz,233.3D0,1)
    P = consts_double(nx,ny,nz,233.3D0,1)

    stepDivCap = dt / Cap
    cx = stepDivCap / Rx
    cy = stepDivCap / Ry
    cz = stepDivCap / Rz
    cc = 1.0 - (2.0*cx + 2.0*cy + 3.0*cz)
    cx = cx / 2
    cy = cy / 2
    cz = cz / 2

    call tic("hotspot3D")
    do i=1,nt
     T = cx*(AXF(T)+AXB(T))+cy*(AYF(T)+AYB(T))+cz*(AZF(T)+AZB(T)) &
         +(dt/Cap)*P + cz*amb_temp
    enddo
    call toc("hotspot3D")
  end subroutine

  subroutine heat_3d()
    implicit none
    type(array) :: T, new_T
    integer :: i
    T = rands(150, 150, 150, dt=OA_DOUBLE)
    new_T = rands(150, 150, 150, dt=OA_DOUBLE)

    do i=1,10
      new_T = AXF(T) + AXB(T) + AYF(T) + AYB(T) + AZF(T) + AZB(T)
      T = new_T
    enddo

call tic("heat")
    do i=1,150
      new_T = AXF(T) + AXB(T) + AYF(T) + AYB(T) + AZF(T) + AZB(T)
      T = new_T
    enddo
call toc("heat")
    !call display(T,"T=")
    !call display(new_T,"new_T=")


  end subroutine

end module

program main
  !use mpi
  use oa_test
  implicit none
include "mpif.h"
  integer :: step
  integer :: i, nt, nx, ny, nz
  ! initialize OpenArray, no split in z-direction
  call oa_init(MPI_COMM_WORLD, [-1, -1, 1])
  ! get option from command line
!  call oa_get_option(nx, "nx", -1)
!  print*, "nx = ", nx
!  call oa_get_option(ny, "ny", -1)
!  print*, "ny = ", ny
!  call oa_get_option(nz, "nz", -1)
!  print*, "nz = ", nz
!  call oa_get_option(nt, "nt", -1)
!  print*, "nt = ", nt
!
  nx = 10
  ny = 10
  nz = 5
  nt = 5
  call array_creation()
  call arithmetic_operation()
  call array_operation()
  call stencil_operation()
  call io_operation()
  call util_operation()

  call continuity(nt, nx, ny, nz)
  call heat_diffusion(nt, nx, ny, nz)
  call hotspot2D(nt, nx, ny, nz)
  call hotspot3D(nt, nx, ny, nz)
  call heat_3d()
  if(get_rank() .eq. 0)call show_timer()
  call oa_finalize()
end program main

