
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
  
  interface
    subroutine c_destroy_array(A) bind(C, name = 'destroy_array')
      type(c_ptr), intent(in) :: ptr
    end subroutine
  end interface

  interface
    subroutine c_destroy_node(A) bind(C, name = 'destroy_node')
      type(c_ptr), intent(in) :: ptr
    end subroutine
  end interface

///:mute
///:set NAME = [['ones'], ['zeros'], ['rands'], ['seqs']]
///:endmute
///:for t in NAME
  interface
    subroutine ${t[0]}$(m, n, k, st, dt, comm, ap) bind(C, name = '${t[0]}$')
      type(c_ptr), intent(in) :: ptr
    end subroutine
  end interface

///:endfor

///:mute
///:set TYPE = [['int', 'integer'], &
       ['float',  'real'], &
       ['double', 'real(kind=8)']]
///:endmute
///:for t in TYPE
///:endfor
  interface
    subroutine 
      use iso_c_binding
    end subroutine
  end interface

  interface
    subroutine 
      use iso_c_binding
    end subroutine
  end interface

  interface
    subroutine 
      use iso_c_binding
    end subroutine
  end interface

  interface
    subroutine 
      use iso_c_binding
    end subroutine
  end interface

  interface
    subroutine 
      use iso_c_binding
    end subroutine
  end interface


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

///:for t in [['int', 'integer'], &
       ['real',  'real'], &
       ['real8', 'real(kind=8)'], &
       ['array', 'type(array)']]
  subroutine create_node_${t}$(B, A)
    type(node), intent(out) :: B
    type(array) :: A

    !call c function to create a node
  end subroutine
///:endfor
  
end module
