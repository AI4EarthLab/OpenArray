#include "config.h"
  

module oa_min_max
  use iso_c_binding
  use oa_type

  interface max
     module procedure max_node_node
     module procedure max_node_array
     module procedure max_array_node
     module procedure max_array_array
  end interface
  interface min
     module procedure min_node_node
     module procedure min_node_array
     module procedure min_array_node
     module procedure min_array_array
  end interface

  interface max
     module procedure max_node
  end interface 
  interface max
     module procedure max_array
  end interface 
  interface max_at
     module procedure max_at_node
  end interface 
  interface max_at
     module procedure max_at_array
  end interface 
  interface min
     module procedure min_node
  end interface 
  interface min
     module procedure min_array
  end interface 
  interface min_at
     module procedure min_at_node
  end interface 
  interface min_at
     module procedure min_at_array
  end interface 
  interface abs_max
     module procedure abs_max_node
  end interface 
  interface abs_max
     module procedure abs_max_array
  end interface 
  interface abs_max_at
     module procedure abs_max_at_node
  end interface 
  interface abs_max_at
     module procedure abs_max_at_array
  end interface 
  interface abs_min
     module procedure abs_min_node
  end interface 
  interface abs_min
     module procedure abs_min_array
  end interface 
  interface abs_min_at
     module procedure abs_min_at_node
  end interface 
  interface abs_min_at
     module procedure abs_min_at_array
  end interface 
  
contains

  function max_node_node(A, B) result(C)
    implicit none
    type(node) :: A
    type(node) :: B
    type(node)::C
     integer:: id_a, id_b






    call c_new_node_op2_simple(TYPE_MAX2, C%id, A%id,  B%id)
 end function
  function max_node_array(A, B) result(C)
    implicit none
    type(node) :: A
    type(array) :: B
    type(node)::C
     integer:: id_a, id_b





    call c_new_node_array_simple(B%ptr, id_b)
    call c_new_node_op2_simple(TYPE_MAX2, C%id, A%id,  id_b)

 end function
  function max_array_node(A, B) result(C)
    implicit none
    type(array) :: A
    type(node) :: B
    type(node)::C
     integer:: id_a, id_b



    call c_new_node_array_simple(A%ptr,id_a)
    call c_new_node_op2_simple(TYPE_MAX2, C%id,id_a,  B%id)


 end function
  function max_array_array(A, B) result(C)
    implicit none
    type(array) :: A
    type(array) :: B
    type(node)::C
     integer:: id_a, id_b


    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_array_simple(B%ptr, id_b)
    call c_new_node_op2_simple(TYPE_MAX2, C%id,id_a,  id_b)




 end function
  function min_node_node(A, B) result(C)
    implicit none
    type(node) :: A
    type(node) :: B
    type(node)::C
     integer:: id_a, id_b






    call c_new_node_op2_simple(TYPE_MIN2, C%id, A%id,  B%id)
 end function
  function min_node_array(A, B) result(C)
    implicit none
    type(node) :: A
    type(array) :: B
    type(node)::C
     integer:: id_a, id_b





    call c_new_node_array_simple(B%ptr, id_b)
    call c_new_node_op2_simple(TYPE_MIN2, C%id, A%id,  id_b)

 end function
  function min_array_node(A, B) result(C)
    implicit none
    type(array) :: A
    type(node) :: B
    type(node)::C
     integer:: id_a, id_b



    call c_new_node_array_simple(A%ptr,id_a)
    call c_new_node_op2_simple(TYPE_MIN2, C%id,id_a,  B%id)


 end function
  function min_array_array(A, B) result(C)
    implicit none
    type(array) :: A
    type(array) :: B
    type(node)::C
     integer:: id_a, id_b


    call c_new_node_array_simple(A%ptr, id_a)
    call c_new_node_array_simple(B%ptr, id_b)
    call c_new_node_op2_simple(TYPE_MIN2, C%id,id_a,  id_b)




 end function


  function max_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_max_simple(id_a, id_b) &
            bind(C, name = "c_new_node_max_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_max_simple(B%id, A%id)

  end function
  function max_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_max_simple(id_a, id_b) &
            bind(C, name = "c_new_node_max_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_max_simple(B%id, id_a)

  end function
  function max_at_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_max_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_max_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_max_at_simple(B%id, A%id)

  end function
  function max_at_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_max_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_max_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_max_at_simple(B%id, id_a)

  end function
  function min_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_min_simple(id_a, id_b) &
            bind(C, name = "c_new_node_min_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_min_simple(B%id, A%id)

  end function
  function min_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_min_simple(id_a, id_b) &
            bind(C, name = "c_new_node_min_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_min_simple(B%id, id_a)

  end function
  function min_at_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_min_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_min_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_min_at_simple(B%id, A%id)

  end function
  function min_at_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_min_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_min_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_min_at_simple(B%id, id_a)

  end function
  function abs_max_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_max_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_max_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_abs_max_simple(B%id, A%id)

  end function
  function abs_max_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_max_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_max_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_abs_max_simple(B%id, id_a)

  end function
  function abs_max_at_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_max_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_max_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_abs_max_at_simple(B%id, A%id)

  end function
  function abs_max_at_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_max_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_max_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_abs_max_at_simple(B%id, id_a)

  end function
  function abs_min_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_min_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_min_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_abs_min_simple(B%id, A%id)

  end function
  function abs_min_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_min_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_min_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_abs_min_simple(B%id, id_a)

  end function
  function abs_min_at_node(A) result(B)
    implicit none
    type(node) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_min_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_min_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    
    call c_new_node_abs_min_at_simple(B%id, A%id)

  end function
  function abs_min_at_array(A) result(B)
    implicit none
    type(array) :: A
    type(node)::B
    integer:: id_a
    interface
       subroutine c_new_node_abs_min_at_simple(id_a, id_b) &
            bind(C, name = "c_new_node_abs_min_at_simple")
         use iso_c_binding
         implicit none
        integer:: id_a,id_b
       end subroutine
    end interface
    
    call c_new_node_array_simple(A%ptr, id_a)
    
    call c_new_node_abs_min_at_simple(B%id, id_a)

  end function

end module

