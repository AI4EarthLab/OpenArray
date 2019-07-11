module oa_init_fnl
  use iso_c_binding
  use oa_type
  
  interface
     subroutine c_init(comm, ps, cmd) &
          bind(C, name="c_init") 
       use iso_c_binding
       integer(c_int), value :: comm
       integer(c_int) :: ps(3)
       character(c_char) :: cmd(*)
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
    character(len=1000) :: cmd

    call get_command(cmd)

    call c_init(comm, ps, string_f2c(cmd))
  end subroutine 

  subroutine oa_finalize()
    call c_finalize()
  end subroutine 

end module oa_init_fnl
