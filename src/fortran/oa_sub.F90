
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
     subroutine c_new_node_sub_node_simple(rx, ry, rz, id_b, id_a) &
          bind(C, name="c_new_node_sub_node_simple")
       use iso_c_binding       
       implicit none
       integer:: id_a, id_b
       integer :: rx(2), ry(2), rz(2)
     end subroutine
  end interface

  interface
     subroutine c_new_node_slice_node_simple(k, res_id, id1) &
          bind(C, name="c_new_node_slice_node_simple")
       use iso_c_binding       
       implicit none
       integer :: k ,res_id, id1
     end subroutine
  end interface

  
  interface sub
     module procedure sub_node_int2
     module procedure sub_node_int
     module procedure sub_node_char
     module procedure sub_array_int2
     module procedure sub_array_int
     module procedure sub_array_char

     module procedure sub_node_int2_int2
     module procedure sub_node_int2_int
     module procedure sub_node_int2_char
     module procedure sub_node_int_int2
     module procedure sub_node_int_int
     module procedure sub_node_int_char
     module procedure sub_node_char_int2
     module procedure sub_node_char_int
     module procedure sub_node_char_char
     module procedure sub_array_int2_int2
     module procedure sub_array_int2_int
     module procedure sub_array_int2_char
     module procedure sub_array_int_int2
     module procedure sub_array_int_int
     module procedure sub_array_int_char
     module procedure sub_array_char_int2
     module procedure sub_array_char_int
     module procedure sub_array_char_char
     
     module procedure sub_node_int2_int2_int2
     module procedure sub_node_int2_int2_int
     module procedure sub_node_int2_int2_char
     module procedure sub_node_int2_int_int2
     module procedure sub_node_int2_int_int
     module procedure sub_node_int2_int_char
     module procedure sub_node_int2_char_int2
     module procedure sub_node_int2_char_int
     module procedure sub_node_int2_char_char
     module procedure sub_node_int_int2_int2
     module procedure sub_node_int_int2_int
     module procedure sub_node_int_int2_char
     module procedure sub_node_int_int_int2
     module procedure sub_node_int_int_int
     module procedure sub_node_int_int_char
     module procedure sub_node_int_char_int2
     module procedure sub_node_int_char_int
     module procedure sub_node_int_char_char
     module procedure sub_node_char_int2_int2
     module procedure sub_node_char_int2_int
     module procedure sub_node_char_int2_char
     module procedure sub_node_char_int_int2
     module procedure sub_node_char_int_int
     module procedure sub_node_char_int_char
     module procedure sub_node_char_char_int2
     module procedure sub_node_char_char_int
     module procedure sub_node_char_char_char
     module procedure sub_array_int2_int2_int2
     module procedure sub_array_int2_int2_int
     module procedure sub_array_int2_int2_char
     module procedure sub_array_int2_int_int2
     module procedure sub_array_int2_int_int
     module procedure sub_array_int2_int_char
     module procedure sub_array_int2_char_int2
     module procedure sub_array_int2_char_int
     module procedure sub_array_int2_char_char
     module procedure sub_array_int_int2_int2
     module procedure sub_array_int_int2_int
     module procedure sub_array_int_int2_char
     module procedure sub_array_int_int_int2
     module procedure sub_array_int_int_int
     module procedure sub_array_int_int_char
     module procedure sub_array_int_char_int2
     module procedure sub_array_int_char_int
     module procedure sub_array_int_char_char
     module procedure sub_array_char_int2_int2
     module procedure sub_array_char_int2_int
     module procedure sub_array_char_int2_char
     module procedure sub_array_char_int_int2
     module procedure sub_array_char_int_int
     module procedure sub_array_char_int_char
     module procedure sub_array_char_char_int2
     module procedure sub_array_char_char_int
     module procedure sub_array_char_char_char
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

  function sub_node_int2(A, rx) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int(A, rx) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char(A, rx) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [0, s(1)]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_array_int2(A, rx) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int(A, rx) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_char(A, rx) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a

    s = shape(A)

    rx1 = [0, s(1)]

    ry1 = [0, s(2)]
    rz1 = [0, s(3)]
    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function


  function sub_node_int2_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_array_int2_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int2_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int2_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx(1) - 1, rx(2)]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_int_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [rx-1, rx]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_char_int2 &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_char_int &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function
  function sub_array_char_char &
       (A, rx, ry) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)

    rx1 = [0, s(1)]


    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, id_a)

  end function


  function sub_node_int2_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int2_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_int_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_node_char_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(node), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id, A%id)

  end function
  function sub_array_int2_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int2_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer, dimension(2) :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx(1)-1,rx(2)]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_int_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [rx-1, rx]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int2_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int2_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int2_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer, dimension(2) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry(1) - 1, ry(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_int_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    integer :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [ry-1, ry]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_char_int2 &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer, dimension(2) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [rz(1) - 1, rz(2)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_char_int &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    integer :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [rz-1, rz]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function
  function sub_array_char_char_char &
       (A, rx, ry, rz) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    character(len=1) :: rx
    character(len=1) :: ry
    character(len=1) :: rz    
    integer :: rx1(2), ry1(2), rz1(2), s(3)
    integer :: id_a
    s = shape(A)
    
    rx1 = [0, s(1)]

    ry1 = [0, s(2)]

    rz1 = [0, s(3)]

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_sub_node_simple(rx1, ry1, rz1, B%id , id_a)

  end function

  function slice(A, k) result(B)
    implicit none
    type(node) :: B
    type(array), intent(in) :: A
    integer :: k
    integer :: rk
    integer :: id_a
    rk = k - 1

    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_slice_node_simple(k, B%id, id_a)

  end function


end module oa_sub
