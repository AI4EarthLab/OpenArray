
module oa_sub
  use iso_c_binding
  use oa_type
  
  ! interface
  !    subroutine c_new_node_sub_array(A, B, rx, ry, rz) &
  !         bind(C, name="c_new_node_sub_array")
  !      use iso_c_binding       
  !      implicit none
  !      type(c_ptr) :: A, B
  !      integer :: rx(2), ry(2), rz(2)
  !    end subroutine
  ! end interface
  
  interface
     subroutine c_new_node_sub_node(A, B, rx, ry, rz) &
          bind(C, name="c_new_node_sub_node")
       use iso_c_binding       
       implicit none
       type(c_ptr) :: A, B
       integer :: rx(2), ry(2), rz(2)
     end subroutine
  end interface

  interface
     subroutine c_new_node_slice_node(A, B, k) &
          bind(C, name="c_new_node_slice_node")
       use iso_c_binding       
       implicit none
       type(c_ptr) :: A, B
       integer :: k
     end subroutine
  end interface

  ///:mute
  ///:set ptype = [['int2', 'integer, dimension(2)'], &
       ['int', 'integer'], &
       ['char', 'character(len=1)']]
  ///:endmute
  
  interface sub
     ///:for t in ['node', 'array']
     ///:for rx in ptype
     module procedure sub_${t}$_${rx[0]}$
     ///:endfor
     ///:endfor

     ///:for t in ['node', 'array']
     ///:for rx in ptype
     ///:for ry in ptype
     module procedure sub_${t}$_${rx[0]}$_${ry[0]}$
     ///:endfor
     ///:endfor
     ///:endfor
     
     ///:for t in ['node', 'array']     
     ///:for rx in ptype
     ///:for ry in ptype
     ///:for rz in ptype  
     module procedure sub_${t}$_${rx[0]}$_${ry[0]}$_${rz[0]}$
     ///:endfor
     ///:endfor
     ///:endfor
     ///:endfor
  end interface sub

  integer, dimension(2), parameter :: RALL = [0, huge(int(1, kind=4))/2] 
contains

  ! function sub_array(A) result(B)
  !   use iso_c_binding
  !   implicit none
  !   type(array), intent(in) :: A
  !   type(array) :: B

    
  !   !write(*, "(Z16.16)") B%ptr
  !   !B%ptr = C_NULL_PTR
  ! end function

  ///:for t in ['node', 'array']
  ///:for rx in ptype
  function sub_${t}$_${rx[0]}$(A, rx) result(B)
    implicit none
    type(${t}$), intent(in) :: A
    type(node) :: B, NA
    ${rx[1]}$ :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)

    s = shape(A)

    ///:if rx[0] == 'int'
    rx1 = [rx-1, rx]
    ///:elif rx[0] == 'int2'
    rx1 = [rx(1) - 1, rx(2)]
    ///:else
    rx1 = [0, s(1)]
    ///:endif

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_sub_node(B%ptr, NA%ptr, rx1, ry1, rz1)
    ///:else
    call c_new_node_sub_node(B%ptr, A%ptr, rx1, ry1, rz1)
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
  end function
  ///:endfor
  ///:endfor


  ///:for t in ['node', 'array']
  ///:for rx in ptype
  ///:for ry in ptype
  function sub_${t}$_${rx[0]}$_${ry[0]}$ &
       (A, rx, ry) result(B)
    implicit none
    type(${t}$), intent(in) :: A
    type(node) :: B, NA
    ${rx[1]}$ :: rx
    ${ry[1]}$ :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)

    s = shape(A)

    ///:if rx[0] == 'int'
    rx1 = [rx-1, rx]
    ///:elif rx[0] == 'int2'
    rx1 = [rx(1) - 1, rx(2)]
    ///:else
    rx1 = [0, s(1)]
    ///:endif


    ///:if ry[0] == 'int'
    ry1 = [ry-1, ry]
    ///:elif ry[0] == 'int2'
    ry1 = [ry(1) - 1, ry(2)]
    ///:else
    ry1 = [0, s(2)]
    ///:endif

    rz1 = [0, s(3)]

    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_sub_node(B%ptr, NA%ptr, rx1, ry1, rz1)
    ///:else
    call c_new_node_sub_node(B%ptr, A%ptr, rx1, ry1, rz1)
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    
  end function
  ///:endfor
  ///:endfor
  ///:endfor


  ///:for t in ['node', 'array']
  ///:for rx in ptype
  ///:for ry in ptype
  ///:for rz in ptype  
  function sub_${t}$_${rx[0]}$_${ry[0]}$_${rz[0]}$ &
       (A, rx, ry, rz) result(B)
    implicit none
    type(${t}$), intent(in) :: A
    type(node) :: B, NA
    ${rx[1]}$ :: rx
    ${ry[1]}$ :: ry
    ${rz[1]}$ :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)

    s = shape(A)
    
    ///:if rx[0] == 'int'
    rx1 = [rx-1, rx]
    ///:elif rx[0] == 'int2'
    rx1 = [rx(1)-1,rx(2)]
    ///:else
    rx1 = [0, s(1)]
    ///:endif

    ///:if ry[0] == 'int'
    ry1 = [ry-1, ry]
    ///:elif ry[0] == 'int2'
    ry1 = [ry(1) - 1, ry(2)]
    ///:else
    ry1 = [0, s(2)]
    ///:endif

    ///:if rz[0] == 'int'
    rz1 = [rz-1, rz]
    ///:elif rz[0] == 'int2'
    rz1 = [rz(1) - 1, rz(2)]
    ///:else
    rz1 = [0, s(3)]
    ///:endif

    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_sub_node(B%ptr, NA%ptr, rx1, ry1, rz1)
    ///:else
    call c_new_node_sub_node(B%ptr, A%ptr, rx1, ry1, rz1)
    ///:endif

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
    
  end function
  ///:endfor
  ///:endfor
  ///:endfor
  ///:endfor

  function slice(A, k) result(B)
    implicit none
    type(array), intent(in) :: A
    type(node) :: B, NA
    integer :: k
    integer :: rk
    rk = k - 1

    call c_new_node_array(NA%ptr, A%ptr)
    call c_new_node_slice_node(B%ptr, NA%ptr, k)

    call set_rvalue(B)
    call try_destroy(A)
    call destroy(NA)
  end function


end module oa_sub
