#include "config.h"
  
  ///:mute
  ///:include "../NodeTypeF.fypp"
  ///:set types = [['double','real(8)', 'scalar'],&
       ['float','real', 'scalar'], &
       ['int','integer', 'scalar'], &
       ['array', 'type(array)', 'array'], &
       ['node', 'type(node)',  'node']]
  ///:endmute

module oa_min_max
  use iso_c_binding
  use oa_type

  ///:for name in ['max', 'min']
  interface ${name}$
     ///:for t1 in ['node', 'array']
     ///:for t2 in ['node', 'array']     
     module procedure ${name}$_${t1}$_${t2}$
     ///:endfor
     ///:endfor
  end interface
  ///:endfor

  ///:for n1 in ['',    'abs_']  
  ///:for n2 in ['max', 'min']
  ///:for n3 in ['',    '_at']
  ///:set name = "{0}{1}{2}".format(n1,n2,n3)
  ///:for t in ['node', 'array']
  interface ${name}$
     module procedure ${name}$_${t}$
  end interface 
  ///:endfor
  ///:endfor
  ///:endfor
  ///:endfor
  
contains

  ///:for name in [['max', 'MAX2'], ['min','MIN2']]
  ///:for t1 in   ['node', 'array']
  ///:for t2 in   ['node', 'array']     
  function ${name[0]}$_${t1}$_${t2}$(A, B) result(C)
    implicit none
    type(${t1}$) :: A
    type(${t2}$) :: B
    type(node)   :: C
    type(node)   :: NA, NB
    
    ///:if t1 == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    ///:set A = 'NA'
    ///:else
    ///:set A = 'A'
    ///:endif

    ///:if t2 == 'array'
    call c_new_node_array(NB%ptr, B%ptr)
    ///:set B = 'NB'
    ///:else
    ///:set B = 'B'
    ///:endif
    
    call c_new_node_op2(C%ptr, TYPE_${name[1]}$, &
         ${A}$%ptr, ${B}$%ptr)

    call set_rvalue(C)
    call try_destroy(A)
    call try_destroy(B)
    call destroy(NA)
    call destroy(NB)
  end function
  ///:endfor
  ///:endfor
  ///:endfor


  ///:for n1 in ['',    'abs_']  
  ///:for n2 in ['max', 'min']
  ///:for n3 in ['',    '_at']
  ///:set name = "{0}{1}{2}".format(n1,n2,n3)
  ///:for t in ['node', 'array']
  function ${name}$_${t}$(A) result(B)
    implicit none
    type(${t}$) :: A
    type(node) :: NA, B

    interface
       subroutine c_new_node_${name}$(A, B) &
            bind(C, name = "c_new_node_${name}$")
         use iso_c_binding
         implicit none
         type(c_ptr) :: A
         type(c_ptr) :: B
       end subroutine
    end interface
    
    ///:if t == 'array'
    call c_new_node_array(NA%ptr, A%ptr)
    ///:set A = 'NA'
    ///:else
    ///:set A = 'A'
    ///:endif

    call c_new_node_${name}$(B%ptr, ${A}$%ptr)

    call set_rvalue(B)

    call try_destroy(A)
    call destroy(NA)

  end function
  ///:endfor
  ///:endfor
  ///:endfor
  ///:endfor

end module
