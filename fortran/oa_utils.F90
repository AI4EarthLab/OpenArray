module oa_utils
  use iso_c_binding

  interface
    subroutine c_get_rank(rank, fcomm) bind(C, name = 'c_get_rank')
      use iso_c_binding
      integer(c_int) :: rank
      integer(c_int), intent(in), VALUE :: fcomm
    end subroutine
  end interface

  interface
     subroutine c_init(comm, ps) &
          bind(C, name="c_init") 
       use iso_c_binding
       integer(c_int), value :: comm
       integer(c_int) :: ps(3)
     end subroutine
  end interface

  interface
     subroutine c_finalize() &
          bind(C, name="c_finalize")
       use iso_c_binding
     end subroutine
  end interface
  
contains

  subroutine oa_init(comm, ps)
    implicit none
    integer :: comm
    integer :: ps(3)

    call c_init(comm, ps)
  end subroutine 

  subroutine oa_finalize()
    call c_finalize()
  end subroutine 
  
  function get_rank(fcomm) result(rank)
    integer(c_int), intent(in) :: fcomm
    integer :: rank

    call c_get_rank(rank, fcomm)
  end function
end module
