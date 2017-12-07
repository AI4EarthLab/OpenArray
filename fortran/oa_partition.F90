
!interface for partition functions
module oa_partition
  interface
     subroutine get_default_procs_shape(shape) &
          bind(C, name = 'get_default_procs_shape')
       use iso_c_binding
       type(c_ptr), VALUE :: shape
     end subroutine
     
     subroutine set_default_procs_shape(shape) &
          bind(C, name = 'set_default_procs_shape')       
       use iso_c_binding
       type(c_ptr), VALUE :: shape
     end subroutine

     subroutine set_auto_procs_shape() &
          bind(C, name = 'set_auto_procs_shape')
       use iso_c_binding
     end subroutine
  end interface
end module 
