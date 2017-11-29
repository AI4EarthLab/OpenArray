module oa_utils
  use iso_c_binding

  interface
    subroutine oa_mpi_init() bind(C, name = 'oa_mpi_init')
    end subroutine
  end interface

  interface
    subroutine oa_mpi_finalize() bind(C, name = 'oa_mpi_finalize')
    end subroutine
  end interface

  interface
    subroutine get_rank(rank, fcomm) bind(C, name = 'get_rank')
      use iso_c_binding
      integer(c_int) :: rank
      integer(c_int), intent(in), VALUE :: fcomm
    end subroutine
  end interface

end module