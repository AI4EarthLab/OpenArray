
module oa_type
  use instrict :: iso_c_binding
  type Array
     C_PTR :: ptr = C_NULL_PTR
   contains
     final :: destroy_array
  end type Array

  type Node
     C_PTR :: ptr = C_NULL_PTR
   contains
     final :: destroy_node
  end type Node
  
contains

  subroutine destroy_array(A)
    type(Array), intent(inout) :: A

    !call c function to destroy array here.
    A%ptr = C_NULL_PTR
  end subroutine

  subroutine destroy_node(A)
    type(Node), intent(inout) :: A

    !call c function to destroy node here.
    A%ptr = C_NULL_PTR
  end subroutine

#:for t in [['int', 'integer'], &
       ['real',  'real'], &
       ['real8', 'real(kind=8)'], &
       ['array', 'type(array)']]
  subroutine create_node_${t}$(B, A)
    type(node), intent(out) :: B
    type(array) :: A

    !call c function to create a node
  end subroutine
#:endif
  
end module
