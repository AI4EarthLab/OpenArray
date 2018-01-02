  ///:include "../NodeTypeF.fypp"

#include "../fortran/config.h"
module oa_test
  use oa_mod
  integer :: m, n, k
  integer(c_int) :: rank, fcomm

///:set types = &
     [['int',   'integer'], &
     [ 'float',  'real'], &
     [ 'double','real(8)']]


contains

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

    ///:for f in ['ones', 'zeros', 'seqs', 'rands']    
    ///:for t in ['OA_INT','OA_FLOAT', 'OA_DOUBLE']
    ///:for s in [0, 1, 2]

    A = ${f}$(m, n, k)
    call display(A, "A = ${f}$(m, n, k)")
    
    A = ${f}$(m, n, k, sw=${s}$)
    call display(A, "${f}$(m, n, k, sw=${s}$)")
    
    A = ${f}$(m, n, k, dt=${t}$)
    call display(A, "${f}$(m, n, k, dt=${t}$)")
    
    A = ${f}$(m, n, k, sw=${s}$, dt=${t}$)
    call display(A, "${f}$(m, n, k, sw=${s}$, dt=${t}$)")
    ///:endfor
    ///:endfor
    ///:endfor
    
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
    type(array) :: A, B, C
    integer :: si
    real :: sr
    real(kind=8) :: sr8
    
    A = ones(m, n, k)
    call display(A, "A")

    B = seqs(m,n,k,1,0) + 1
    call display(B, "B")

    !array with array
    ///:for o in ['+', '-', '*', '/']
    C = A ${o}$ B
    call display(C, "A ${o}$ B = ")
    ///:endfor

    si  = 2;
    sr  = 3.0;
    sr8 = 4.0_8;

    !scalar with array
    ///:for o in ['+', '-', '*', '/']
    ///:for s in [['si','2'], ['sr','3.0'], ['sr8','4.0_8']]
    C = ${s[0]}$ ${o}$ B
    call display(C, "${s[1]}$ ${o}$ B = ")
    ///:endfor
    ///:endfor

    !array with scalar
    ///:for o in ['+', '-', '*', '/']
    ///:for s in [['si','2'], ['sr','3.0'], ['sr8','4.0_8']]
    C = B ${o}$ ${s[0]}$
    call display(C, "B ${o}$ ${s[1]}$ = ")
    ///:endfor
    ///:endfor
    
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

    ///:set os = ['>', '>=', '<', '<=', '==', '/=']
    ///:set ts = [['si','2'], ['sr','3.0'], ['sr8','4.0_8']]
    
    !array with array
    ///:for o in os
    C = A ${o}$ B
    call display(C, "A ${o}$ B = ")
    ///:endfor

    si  = 2;
    sr  = 3.0;
    sr8 = 4.0_8;

    !scalar with array
    ///:for o in os
    ///:for s in ts
    C = ${s[0]}$ ${o}$ B
    call display(C, "${s[1]}$ ${o}$ B = ")
    ///:endfor
    ///:endfor

    !array with scalar
    ///:for o in os
    ///:for s in ts
    C = B ${o}$ ${s[0]}$
    call display(C, "B ${o}$ ${s[1]}$ = ")
    ///:endfor
    ///:endfor
    
  end subroutine


  subroutine test_math()
    type(array) :: A, B, C, D
    integer :: i

    A = rands(m, n, k, dt=OA_DOUBLE)
    B = rands(m, n, k, dt=OA_DOUBLE)
    C = rands(m, n, k, dt=OA_DOUBLE)
    D = rands(m, n, k, dt=OA_DOUBLE)
    
    ///:for op in [o for o in L if o[3] == 'C']


    D = ${op[2]}$(C)

    
    call display(D, "${op[2]}$(C) = ")
    
    D = ${op[2]}$(C*0.5)
    call display(D, "${op[2]}$(C*0.5) = ")
    
    ///:endfor

    do i = 0, 10000
       FSET(A, B + C + D)
    end do

  end subroutine

  subroutine test_sub()
    implicit none
    type(array) :: A, B, C

    A = seqs(10, 8, 2);
    call display(A, "A = ")

    !sub index is from 1 
    ///:for z in [[1,1],[1, 2]]
    ///:for y in [[1,1],[1,3],[2,5]]
    ///:for x in [[1,1],[2,4],[6,10]]
    B = sub(A, [${x[0]}$, ${x[1]}$], &
         [${y[0]}$,${y[1]}$],[${z[0]}$,${z[1]}$])
    
    call display(B, "sub(A, [${x[0]}$, ${x[1]}$], &
         &[${y[0]}$,${y[1]}$],[${z[0]}$,${z[1]}$]) = ")    
    ///:endfor
    ///:endfor
    ///:endfor

    B = sub(A, [1,3],[1,8],[1,2])
    call display(B, "sub(A, [1,3],[1,8],[1,2]) = ")
    
    B = sub(A, [1,3])
    call display(B, "sub(A, [1,3]) = ")

    B = sub(A, 1);
    call display(B, "sub(A, 1) = ")

    B = sub(A, 1, 2, 2);
    call display(B, "sub(A, 1, 2, 2) = ")

    B = sub(A*2.0, [1,3], 2, ':');
    call display(B, "sub(A, [1,3], 2, ':') = ")
    
  end subroutine

  subroutine test_rep()
    implicit none
    type(array) :: A, B, C
    type(node) :: nn
    
    A = seqs(m, n, k)

    ///:for dx in [1,2,3]
    ///:for dy in [1,2,3]
    ///:for dz in [1,2]

    B = rep(A, ${dx}$, ${dy}$, ${dz}$)
    call display(B, "rep(A, ${dx}$, ${dy}$, ${dz}$) = ")
    
    C = rep(A+A, ${dx}$, ${dy}$, ${dz}$)
    call display(C, "rep(A+A, ${dx}$, ${dy}$, ${dz}$) = ")
    
    ///:endfor
    ///:endfor
    ///:endfor
  end subroutine
  
  subroutine test_sum()
    implicit none
    type(array) :: A, B, C, D

    ///:for t in ['OA_INT', 'OA_DOUBLE', 'OA_FLOAT']
    A = ones(m, n, k, dt = ${t}$)
    B = seqs(m, n, k, dt = ${t}$)

    call display(A, "A = ")
    call display(B, "B = ")
    
    ///:for d in [1, 2, 3]
    C = csum(A, ${d}$)
    call display(C, "csum(A, ${d}$) = ")

    C = csum(B, ${d}$)
    call display(C, "csum(B, ${d}$) = ")

    C = csum(A+B, ${d}$)
    call display(C, "csum(A+B, ${d}$) = ")
    
    D = sum(B, ${d}$)
    call display(D, "sum(B, ${d}$) = ")
    
    ///:endfor
    ///:endfor
    
  end subroutine test_sum

  subroutine test_min_max()
    implicit none
    type(array) :: A, B, C, D

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
    
  end subroutine
  
  subroutine test_operator()
    implicit none
    type(array) :: A, B, C
    
    !call grid_init('C', dx, dy, dz)

    A = rands(4,4,4) !seqs(4, 4, 4) + 1
    call display(A, "A = ")

    ///:for o in ['AXB', 'AXF', 'AYB', 'AYF', 'AZB', 'AZF', &
         'DXB', 'DXF', 'DYB', 'DYF', 'DZB', 'DZF', 'DXC', &
         'DYC', 'DZC']

    B = ${o}$(A)
    call display(A, "A = ")
    call display(B, "${o}$(A)")
    
    ///:endfor

    A = rands(m, n, 1)
    ///:for o in ['AXB', 'AXF', 'DXB', 'DXF', 'DXC']
    B = ${o}$(A)
    call display(B, "${o}$(A)")
    ///:endfor

    A = rands(1, n, k)
    ///:for o in ['AZB', 'AZF', 'DZB', 'DZF', 'DZC']

    B = ${o}$(A)
    call display(A, "A = ")
    call display(B, "${o}$(A)")
    ///:endfor

    A = rands(1, 1, k)
    ///:for o in ['AZB', 'AZF', 'DZB', 'DZF', 'DZC']
    B = ${o}$(A)
    call display(A, "A = ")
    call display(B, "${o}$(A)")
    ///:endfor

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

    ///:for d in fdim
    ///:for t in scalar_dtype
    ${t[1]}$, allocatable :: farr_${t[0]}$${d[0]}$(${d[1]}$)
    ///:endfor
    ///:endfor
    
    val1 = 9
    val2 = 9.900
    val3 = 9.9900


    ///:for t in [['OA_INT','val1'], &
        ['OA_FLOAT','val2'], &
        ['OA_DOUBLE','val3']]

    A = seqs(8, 8, 4, dt = ${t[0]}$);
    !call set(A, [3,5], [2,5],[1,3], ${t[1]}$)
    call set(sub(A,[3,5],[2,5],[1,3]), ${t[1]}$)
    !call set_with_const(A, [3,5], [2,5],[1,3], ${t[1]}$)
    call display(A, "A = ")

    ///:endfor


    ///:for t in [['OA_INT','val1'], &
        ['OA_FLOAT','val2'], &
        ['OA_DOUBLE','val3']]

    A = seqs(8, 8, 4, dt = ${t[0]}$);
    B = seqs(3, 4, 3, dt = ${t[0]}$);
    !call set(A, [3,5], [2,5],[1,3], B)
    call set(sub(A, [3,5], [2,5],[1,3]), B)
    
    call display(A, "A = ")

    ///:endfor


    ///:for t in [['OA_INT','val1'], &
        ['OA_FLOAT','val2'], &
        ['OA_DOUBLE','val3']]

    A = seqs(8, 8, 4, dt = ${t[0]}$);
    B = ones(8, 8, 4, dt = ${t[0]}$);
    call set(sub(A, [3,5], [2,5],[1,3]), &
         sub(B, [1,3], [1,4], [1,3]))
    call display(A, "A = ")

    ///:endfor

    A = 1.1;
    call display(A, "A = ")

    call set(sub(A,[3,5],[2,5],[1,3]), 2.1_8);
    call display(A, "A = ")

    ///:set fdim1 = [[1,'10',':'],[2,'5,5',':,:'],&
         [3,'5,4,3',':,:,:']]
    ///:for d in fdim1
    ///:for t in scalar_dtype
    allocate(farr_${t[0]}$${d[0]}$(${d[1]}$))
    farr_${t[0]}$${d[0]}$(${d[2]}$) = 10
    A = farr_${t[0]}$${d[0]}$
    call display(A, "A = ")
    ///:endfor
    ///:endfor

    A = seqs(10,10,10)
    call set(sub(A, [1,10],[1,1],[1,1]),  farr_double1)
    call display(A, "A = ")
    
    A = seqs(10,10,10)
    call set(sub(A, [1,5], [1,5],[1,1]),  farr_double2)
    call display(A, "A = ")
    
    A = seqs(10,10,10)
    call set(sub(A, [1,5], [1,4],[1,3]),  farr_double3)
    call display(A, "A = ")

    A = seqs(10,10,10)
    deallocate(farr_double3)
    allocate(farr_double3(1,5,5))
    farr_double3 = 10
    call set(sub(A,1,[1,5],[1,5]), farr_double3)
    call display(A, "A = ")

    A = seqs(10,10,10)
    deallocate(farr_double3)
    allocate(farr_double3(1,1,10))
    farr_double3 = 10
    call set(sub(A,1,1,[1,10]), farr_double3)
    call display(A, "A = ")

    call set(val3, sub(a, 1, 1, 1))
    print*, "val3 = ", val3

    call set(val3, sub(a, 1, 2, 1))
    print*, "val3 = ", val3
    
    call set(val3, sub(a+a, 3, 2, 2))
    print*, "val3 = ", val3
    
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
    
    ///:for o in ['AXB', 'AXF', 'AYB', 'AYF', &
         'DXB', 'DXF', 'DYB', 'DYF']
    B = ${o}$(A) 
    call display(B, "${o}$(A) = ")
    ///:endfor

  end subroutine

  subroutine test_shift()
    implicit none
    type(array) :: A
    type(array) :: B
    
    A = seqs(6, 8, 3)
    ! call display(A, "A")

    B = shift(A, 1)

    call display(B, "B = ")
    
    B = shift(A, -1)
    call display(B, "B = ")
    
    ! B = shift(A, 1, 2,0)
    
    ! print*, "after shifted"
    ! call display(A, "A")
  end subroutine


  subroutine test_io()
    implicit none
    type(array) :: A
    type(array) :: B
    
    A = seqs(6, 8, 3)

    call save(A, "A.nc", "data")

    B = load("A.nc", "data")

    call display(B, "B = ")
  end subroutine

  subroutine test_get_ptr()
    implicit none
    type(array) :: A
    type(array) :: B
    real(kind=8), pointer :: p(:,:,:)
    integer :: i, j, k, s(3)
    
    A = ones(4, 3, 2)

    call display(A, "A = ")
    
    p => get_buffer_double(A)
    !print*, "shape(p) = ", shape(p)

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
end module
