  ///:include "NodeTypeF.fypp"
  
module oa_test
  use oa_mod
  integer :: m, n, k
  integer(c_int) :: rank, fcomm
contains

  subroutine test_init(mm, nn, kk, comm)
    integer :: mm, nn, kk
    integer(c_int) :: comm
    m = mm
    n = nn
    k = kk

    fcomm = comm
    rank = get_rank(fcomm)
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

    call c_new_seqs_scalar_node_int(X%ptr, 1, fcomm)
    call display(X, "X")

    call c_new_seqs_scalar_node_float(Y%ptr, 2.1, fcomm)
    call display(Y, "Y")

    call c_new_seqs_scalar_node_double(Z%ptr, 3.1_8, fcomm)
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

    C = rands(m, n, k, dt=OA_DOUBLE)
    
    ///:for op in [o for o in L if o[3] == 'C']

    D = ${op[2]}$(C)
    call display(D, "${op[2]}$(C) = ")
    
    D = ${op[2]}$(C*0.5)
    call display(D, "${op[2]}$(C*0.5) = ")
    
    ///:endfor
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

    B = sub(A, [1,3])
    call display(B, "sub(A, [1,3]) = ")

    B = sub(A, 1);
    call display(B, "sub(A, 1) = ")

    B = sub(A, 1, 2, 2);
    call display(B, "sub(A, 1, 2, 2) = ")

    B = sub(A, [1,3], 2, ':');
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
  
  subroutine test_set()
    
  end subroutine
  
  subroutine test_parition()
    integer :: A(3), B(3), C(3)

    call get_procs_shape(A)
    print*, "A = ", A

    B = [1,2,3]
    call set_procs_shape(B)
    call get_procs_shape(C)
    print*, "C = ", C

    call set_auto_procs_shape()
    call get_procs_shape(C)
    print*, "C = ", C

  end subroutine test_parition

end module
