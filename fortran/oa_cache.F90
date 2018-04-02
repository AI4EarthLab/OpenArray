
module oa_cache
  use iso_c_binding
  use oa_type

  type(node) :: tmp_node__
  character(len=1000) :: tmp_node_key__
  
contains

  subroutine gen_node_key__(file, line)
    use iso_c_binding
    implicit none
    
    character(len=*) :: file
    integer :: line
    !character(len=len(file)+8) :: buf
    !character(kind=c_char) :: res(*)
    
    !write(tmp_node_key__, "(A, A, I0)") &
    !     file,":",line
    
    !res = string_f2c(buf)
  end subroutine

  subroutine find_node__()
    use iso_c_binding
    implicit none

    interface
       subroutine c_find_node(node, key) &
            bind(C, name = "c_find_node")
         use iso_c_binding
         implicit none
         type(c_ptr) :: node
         character(kind=c_char) :: key(*)
       end  subroutine
    end interface
    
    ! character(kind=c_char) :: key(*)
    ! type(node) :: res

    call c_find_node(tmp_node__%ptr, &
         string_f2c(tmp_node_key__))

  end subroutine

  function is_valid__() result(res)
    implicit none

    interface
       subroutine c_is_null(p, i) &
            bind(C, name="c_is_null")
         use iso_c_binding
         type(c_ptr) :: p
         integer, intent(out) :: i
       end subroutine c_is_null
    end interface
    
    logical :: res
    integer :: i

    call c_is_null(tmp_node__%ptr, i)
    
    res = (i /= 0);
    ! if(.not. res) print*, "not found node"
    ! if(res) print*, "found node"    
  end function

  subroutine cache_node__()
    implicit none

    interface
       subroutine c_cache_node(node_ptr, key_char) &
            bind(C, name = "c_cache_node")
         use iso_c_binding
         implicit none
         type(c_ptr) :: node_ptr
         character(kind=c_char) :: key_char(*)
       end subroutine
    end interface
    
    ! type(node) :: node
    ! character(kind=c_char) :: key(*)

    call c_cache_node(tmp_node__%ptr, &
         string_f2c(tmp_node_key__))

  end subroutine
end module
