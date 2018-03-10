///:mute
///:set types = &
     [['int',    'integer', 'integer(c_int)'], &
     [ 'float',  'real',    'real(c_float)'], &
     [ 'double', 'real(8)', 'real(c_double)']]
///:endmute

module oa_set
  use iso_c_binding
  use oa_type

  interface set
     ///:for type1 in types
     module procedure set_ref_const_${type1[0]}$
     module procedure set_array_const_${type1[0]}$
     ///:endfor
     module procedure set_ref_array
     module procedure set_ref_ref
     
     ///:for dim in ['1d','2d','3d']
     ///:for type in types
     module procedure set_array_farray_${type[0]}$_${dim}$
     module procedure set_ref_farray_${type[0]}$_${dim}$

     module procedure set_farray_array_${type[0]}$_${dim}$
     module procedure set_farray_node_${type[0]}$_${dim}$
     ///:endfor
     ///:endfor

     !set array/node to a fortran scalar
     ///:for type1 in types
     ///:for type2 in ['node', 'array']
     module procedure set_${type1[0]}$_${type2}$
     ///:endfor
     ///:endfor
  end interface set

  interface assignment(=)
     ///:for type1 in types
     module procedure set_array_const_${type1[0]}$
     ///:endfor

     !set fortran array to array object
     ///:for dim in ['1d','2d','3d']
     ///:for type in types
     module procedure set_array_farray_${type[0]}$_${dim}$
     ///:endfor
     ///:endfor

     !set array/node to a fortran array
     ///:for dim in ['1d','2d','3d']
     ///:for type in types
     module procedure set_farray_array_${type[0]}$_${dim}$
     module procedure set_farray_node_${type[0]}$_${dim}$
     ///:endfor
     ///:endfor

     !set array/node to a fortran scalar
     ///:for type1 in types
     ///:for type2 in ['node', 'array']
     module procedure set_${type1[0]}$_${type2}$
     ///:endfor
     ///:endfor     
  end interface assignment(=)
  
contains

  ///:for type1 in types
  subroutine set_ref_const_${type1[0]}$(A, val)
    implicit none

    interface
       subroutine c_set_ref_const_${type1[0]}$(A,  val) &
            bind(C, name="c_set_ref_const_${type1[0]}$")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
       end subroutine
    end interface

    type(node), intent(in) :: A
    ${type1[1]}$ :: val

    call c_set_ref_const_${type1[0]}$(A%ptr, val)

    call try_destroy(A)
  end subroutine


  subroutine set_array_const_${type1[0]}$(A, val)
    implicit none

    interface
       subroutine c_set_array_const_${type1[0]}$(A,  val) &
            bind(C, name="c_set_array_const_${type1[0]}$")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         ${type1[1]}$, value :: val
       end subroutine
    end interface

    type(array), intent(inout) :: A
    ${type1[1]}$, intent(in) :: val
    !print*, "set array const..."
    call c_set_array_const_${type1[0]}$(A%ptr, val)

    call try_destroy(A)

  end subroutine
  ///:endfor

   subroutine set_ref_array(A, B)
     implicit none
     
     interface
        subroutine c_set_ref_array(a, b) &
             bind(C, name="c_set_ref_array")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
        end subroutine
     end interface
     
    type(node), intent(in) :: A
    type(array) :: B
    
    call c_set_ref_array(A%ptr, B%ptr)

    call try_destroy(A)
    call try_destroy(B)
  end subroutine
  
  subroutine set_ref_ref(A, B)
    implicit none
    interface
        subroutine c_set_ref_ref(a, b) &
             bind(C, name="c_set_ref_ref")
          use iso_c_binding
          implicit none
          type(c_ptr) :: a, b
        end subroutine
     end interface

    type(node), intent(in) :: A, B

    call c_set_ref_ref(A%ptr, B%ptr)
    
    call try_destroy(A)
    call try_destroy(B)
  end subroutine

  

  !> set_ref_farray
  ///:for dim in [[1,':'],[2,':,:'],[3,':,:,:']]
  ///:for type in types
  ///:for at in [['ref','node'], ['array', 'array']]
  subroutine set_${at[0]}$_farray_${type[0]}$_${dim[0]}$d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_${at[0]}$_farray_${type[0]}$(A, arr, s) &
            bind(C, name = 'c_set_${at[0]}$_farray_${type[0]}$')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    ///:if at[0] == 'array'
    type(${at[1]}$), intent(inout) :: A
    ///:else
    type(${at[1]}$), intent(in) :: A
    ///:endif
    ${type[1]}$, dimension(${dim[1]}$), target, intent(in) :: B
    
    ! ${type[1]}$, target, allocatable, &
    !      dimension(${dim[1]}$) :: B
    integer :: s(${dim[0]}$), s3(3)

    s = shape(B)
    ///:if dim[0] == 1
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1
    ///:elif dim[0] == 2
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1
    ///:elif dim[0] == 3
    s3 = s
    ///:endif

    call c_set_${at[0]}$_farray_${type[0]}$(A%ptr, c_loc(B), s3)

    call try_destroy(A)

  end subroutine

  ///:endfor
  ///:endfor
  ///:endfor  

  ///:for dim in [[1,':'],[2,':,:'],[3,':,:,:']]
  ///:for type in types
  ///:for at in [['node','node'], ['array', 'array']]
  subroutine set_farray_${at[0]}$_${type[0]}$_${dim[0]}$d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_${at[0]}$_${type[0]}$(A, B, s) &
            bind(C, name = 'c_set_farray_${at[0]}$_${type[0]}$')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    ${type[1]}$, target,&
         dimension(${dim[1]}$), intent(out) :: A

    type(${at[1]}$), intent(in) :: B
    
    integer :: s(${dim[0]}$), s3(3)

    s = shape(A)
    ///:if dim[0] == 1
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1
    ///:elif dim[0] == 2
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1
    ///:elif dim[0] == 3
    s3 = s
    ///:endif

    call c_set_farray_${at[0]}$_${type[0]}$(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  ///:endfor
  ///:endfor
  ///:endfor  


  
  ///:for type1 in types
  ///:for type2 in ['node', 'array']
  subroutine set_${type1[0]}$_${type2}$(A, B)
    use iso_c_binding
    implicit none
    ${type1[1]}$, intent(inout) :: A
    type(${type2}$),intent(in) :: B

    interface
       subroutine c_set_${type1[0]}$_${type2}$(A, B) &
            bind(C, name = "c_set_${type1[0]}$_${type2}$")
         use iso_c_binding
         implicit none
         type(c_ptr) :: B
         ${type1[2]}$, intent(out) :: A
       end subroutine
    end interface

    call c_set_${type1[0]}$_${type2}$(A, B%ptr)
    
    call try_destroy(B)
  end subroutine
  ///:endfor
  ///:endfor
  
end module oa_set
