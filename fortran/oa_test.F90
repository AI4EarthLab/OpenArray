
program test
  use oa_type
  use oa_ops

contains
  subroutine test_basic()
    type(array) :: A, B, C

    D = A + B
  end subroutine
end program test
