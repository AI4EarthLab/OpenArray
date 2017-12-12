  ///:mute
  ///:include "NodeTypeF.fypp"
  ///:set types = [['double','real(8)', 'scalar'],&
       ['float','real', 'scalar'], &
       ['int','integer', 'scalar'], &
       ['array', 'type(array)', 'array'], &
       ['node', 'type(node)',  'node']]
  ///:endmute

#include "config.h"

  module oa_ops
    use iso_c_binding
    use oa_type

    ///:for t in types
    ///:if t[2] == 'scalar'
    ! interface
    !   subroutine c_new_seqs_scalar_node_${t[0]}$(ptr, val, comm) &
    !        bind(C, name = 'new_seqs_scalar_node_${t[0]}$')
    !     use iso_c_binding
    !     type(c_ptr), intent(inout) :: ptr
    !     ${t[1]}$, intent(in), VALUE :: val
    !     integer(c_int), intent(in), VALUE :: comm
    !   end subroutine
    ! end interface
    ///:endif
    ///:endfor

    ! interface
    !   subroutine c_new_node_array(A, B) bind(C, name='new_node_array')
    !     type(c_ptr) :: A
    !     type(c_ptr) :: B
    !   end subroutine
    ! end interface

    ! interface
    !   subroutine c_new_node_op2(A, nodetype, U, V) &
    !       bind(C, name='new_node_op2')
    !     type(c_ptr), intent(inout) :: A
    !     type(c_ptr), intent(in) :: U, V 
    !     integer(c_int), intent(in), VALUE :: nodetype
    !   end subroutine
    ! end interface

    ! interface
    !   subroutine c_new_node_op1(A, nodetype, U) &
    !       bind(C, name='new_node_op1')
    !     type(c_ptr), intent(inout) :: A
    !     type(c_ptr), intent(in) :: U 
    !     integer(c_int), intent(in), VALUE :: nodetype
    !   end subroutine
    ! end interface

    interface assignment(=)
       module procedure array_assign_array
       module procedure node_assign_node
       module procedure node_assign_array
    end interface assignment(=)

    interface
       subroutine c_array_assign_array(A, B, pa, pb) &
            bind(C, name='c_array_assign_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
         integer(c_int), intent(inout) :: pa
         integer(c_int), intent(in) :: pb
       end subroutine
    end interface

    interface
       subroutine c_node_assign_node(A, B) &
            bind(C, name='c_node_assign_node')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
       end subroutine
    end interface

    interface
       subroutine c_node_assign_array(A, B) &
            bind(C, name='c_node_assign_array')
         use iso_c_binding
         type(c_ptr), intent(inout) :: A
         type(c_ptr), intent(in) :: B
       end subroutine
    end interface

    ///:for op in [o for o in L if o[3] == 'A' or o[3] == 'B']
    interface operator (${op[2]}$)
       ///:for type1 in types
       ///:for type2 in types
       ///:if not (type1[2] == 'scalar' and type2[2] == 'scalar')
       module procedure ops_${type1[0]}$_${op[1]}$_${type2[0]}$
       ///:endif
       ///:endfor
       ///:endfor
    end interface operator (${op[2]}$)
    ///:endfor

    ///:for op in [o for o in L if o[3] == 'C']
    ///:set b = 'operator ({0})'.format(op[2]) &
         if (op[2] in ['+', '-']) else op[2]
    interface ${b}$ 
       ///:for type1 in types
       ///:if not (type1[2] == 'scalar')
       module procedure ops_${op[1]}$_${type1[2]}$
       ///:endif
       ///:endfor
    end interface ${b}$ 
    ///:endfor
    
  contains

    ///:for e in [x for x in L if x[3] == 'A' or x[3] == 'B']
    ///:set op = e[1]
    ///:for type1 in types
    ///:for type2 in types
    ///:if not (type1[2] == 'scalar' and type2[2] == 'scalar')
    function ops_${type1[0]}$_${op}$_${type2[0]}$(A, B) result(res)
      implicit none       
      ${type1[1]}$, intent(in) :: A
      ${type2[1]}$, intent(in) :: B
      ///:if type1[0] != 'node'
      type(node) :: C
      ///:endif
      ///:if type2[0] != 'node'
      type(node) :: D
      ///:endif
      type(node) :: res

      ///:if type1[0] == 'node'
      ///:set AC = 'A'
      ///:else
      ///:if type1[2] == 'scalar'
      call c_new_seqs_scalar_node_${type1[0]}$(C%ptr, A, MPI_COMM_SELF)
      ///:else
      call c_new_node_array(C%ptr, A%ptr)
      ///:endif
      ///:set AC = 'C'
      ///:endif

      ///:if type2[0] == 'node'
      ///:set BD = 'B'
      ///:else
      ///:if type2[2] == 'scalar'
      call c_new_seqs_scalar_node_${type2[0]}$(D%ptr, B, MPI_COMM_SELF)
      ///:else
      call c_new_node_array(D%ptr, B%ptr)
      ///:endif
      ///:set BD = 'D'
      ///:endif

      call c_new_node_op2(res%ptr, ${e[0]}$, ${AC}$%ptr, ${BD}$%ptr)

    end function

    ///:endif
    ///:endfor
    ///:endfor
    ///:endfor


    ///:for e in [x for x in L if x[3] == 'C']
    ///:set op = e[1]
    ///:for type1 in types
    ///:if not (type1[2] == 'scalar')
    function ops_${op}$_${type1[2]}$(A) result(res)
      implicit none       
      ${type1[1]}$, intent(in) :: A
      type(node) :: res
      ///:if type1[0] != 'node'
      type(node) :: C
      ///:endif
      ///:if type1[0] == 'node'
      ///:set AC = 'A'
      ///:else
      call c_new_node_array(C%ptr, A%ptr)
      ///:set AC = 'C'
      ///:endif
      call c_new_node_op1(res%ptr, ${e[0]}$, ${AC}$%ptr)

    end function

    ///:endif
    ///:endfor
    ///:endfor

    subroutine array_assign_array(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(array), intent(in) :: B

      call c_array_assign_array(A%ptr, B%ptr, A%lr, B%lr)
      
      !print *, A%ptr, B%ptr, A%lr, B%lr
    end subroutine

    subroutine node_assign_node(A, B)
      implicit none
      type(node), intent(inout) :: A
      type(node), intent(in) :: B
      call c_node_assign_node(A%ptr, B%ptr)
      !print *, A%ptr, B%ptr, A%lr, B%lr
    end subroutine

    subroutine node_assign_array(A, B)
      implicit none
      type(array), intent(inout) :: A
      type(node), intent(in) :: B
      call c_node_assign_array(A%ptr, B%ptr)
    end subroutine

  end module
