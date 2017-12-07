
program test
  use oa_type
  use oa_ops
  use oa_interface
contains
  subroutine test_basic()
    type(array) :: A, B, C

    D = A + B
  end subroutine

  subroutine test_parition()
    integer :: A(3), B(3), C(3)

    call get_default_procs_shape(A)
    print*, "A = ", A

    B(3) = [1,2,3]
    call set_default_procs_shape(B)
    call get_default_procs_shape(C)
    print*, "C = ", C

    call set_auto_procs_shape()
    call get_default_procs_shape(C)
    print*, "C = ", C
    
  end subroutine test_parition
  
end program test
