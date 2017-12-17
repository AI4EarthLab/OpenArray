///:mute
///:set types = &
     [['int',   'integer'], &
     [ 'float',  'real'], &
     [ 'double','real(8)']]
///:endmute

module oa_set
  use iso_c_binding
  use oa_type
  interface
     ///:for type1 in types
     subroutine c_set_with_const_${type1[0]}$(A, rx, ry, rz, val) &
          bind(C, name="c_set_with_const_${type1[0]}$")
       use iso_c_binding       
       implicit none
       type(c_ptr) :: A
       integer :: rx(2), ry(2), rz(2)
       ${type1[1]}$ :: val
     end subroutine
     ///:endfor
  end interface

  interface
     subroutine c_set1(A, rx, ry, rz, B) &
          bind(C, name="c_set1")
       use iso_c_binding       
       implicit none
       type(c_ptr) :: A, B
       integer :: rx(2), ry(2), rz(2)
     end subroutine
  end interface

  interface
     subroutine c_set2(A, rx, ry, rz, B, sx, sy, sz) &
          bind(C, name="c_set2")
       use iso_c_binding       
       implicit none
       type(c_ptr) :: A, B
       integer :: rx(2), ry(2), rz(2)
       integer :: sx(2), sy(2), sz(2)
     end subroutine
  end interface

 
!  interface set_with_const
!     ///:for type1 in types
!     module procedure set_with_const_${type1[0]}$
!     ///:endfor
!  end interface set_with_const

  interface set
     ///:for type1 in types
     module procedure set_with_const_${type1[0]}$
     ///:endfor
     module procedure set1
     module procedure set2
  end interface set

    
contains
  ///:for type1 in types
  subroutine set_with_const_${type1[0]}$(A, rx, ry, rz, val)
    implicit none
    type(array), intent(inout) :: A
    integer, dimension(2) :: rx, ry, rz
    integer :: rx1(2), ry1(2), rz1(2)
    ${type1[1]}$ :: val

    rx1 = rx - 1
    ry1 = ry - 1
    rz1 = rz - 1
          
    call c_set_with_const_${type1[0]}$(A%ptr, rx1, ry1, rz1, val)
             
  end subroutine
  ///:endfor

 subroutine set1(A, rx, ry, rz, B)
    implicit none
    type(array), intent(inout) :: A, B
    integer, dimension(2) :: rx, ry, rz
    integer :: rx1(2), ry1(2), rz1(2)

    rx1 = rx - 1
    ry1 = ry - 1
    rz1 = rz - 1
          
    call c_set1(A%ptr, rx1, ry1, rz1, B%ptr)
             
  end subroutine

  subroutine set2(A, rx, ry, rz, B, sx, sy, sz)
    implicit none
    type(array), intent(inout) :: A, B
    integer, dimension(2) :: rx, ry, rz
    integer, dimension(2) :: sx, sy, sz
    integer :: rx1(2), ry1(2), rz1(2)
    integer :: sx1(2), sy1(2), sz1(2)

    rx1 = rx - 1
    ry1 = ry - 1
    rz1 = rz - 1

    sx1 = sx - 1
    sy1 = sy - 1
    sz1 = sz - 1
          
    call c_set2(A%ptr, rx1, ry1, rz1, B%ptr, sx1, sy1, sz1)
             
  end subroutine




end module oa_set
