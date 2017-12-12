
!interface for partition functions
module oa_partition
  interface
     subroutine c_get_default_procs_shape(shape) &
          bind(C, name = 'c_get_default_procs_shape')
       use iso_c_binding
       integer :: shape(3)
     end subroutine
     
     subroutine c_set_default_procs_shape(shape) &
          bind(C, name = 'c_set_default_procs_shape')       
       use iso_c_binding
       integer :: shape(3) 
     end subroutine

     subroutine c_set_auto_procs_shape() &
          bind(C, name = 'c_set_auto_procs_shape')
       use iso_c_binding
     end subroutine
  end interface

contains
  subroutine set_procs_shape(shape)
    integer :: shape(3)
    call c_set_default_procs_shape(shape)
  end subroutine 

  subroutine get_procs_shape(shape)
    integer, intent(out) :: shape(3)
    call c_get_default_procs_shape(shape)
  end subroutine 

  subroutine set_auto_procs_shape()
    call c_set_auto_procs_shape()
  end subroutine 

end module 
