///:mute
///:set types = &
     [['int',   'integer'], &
     [ 'float',  'real'], &
     [ 'double','real(8)']]
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
     ///:endfor
     ///:endfor
  end interface set

  interface assignment(=)
     ///:for type1 in types
     module procedure set_array_const_${type1[0]}$
     ///:endfor

     ///:for dim in ['1d','2d','3d']
     ///:for type in types
     module procedure set_array_farray_${type[0]}$_${dim}$
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
    print*, "set array const..."
    call c_set_array_const_${type1[0]}$(A%ptr, val)

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
             
  end subroutine

  
  ///:for dim in [[1,':'],[2,':,:'],[3,':,:,:']]
  ///:for type in types
  subroutine set_array_farray_${type[0]}$_${dim[0]}$d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_${type[0]}$(A, arr, s) &
            bind(C, name = 'c_set_array_farray_${type[0]}$')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    ${type[1]}$, target, &
         dimension(${dim[1]}$), intent(in) :: B
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

    call c_set_array_farray_${type[0]}$(A%ptr, c_loc(B), s3)
  end subroutine
  
  !module procedure set_ref_farray_${type[0]}$_${dim}$
  ///:endfor
  ///:endfor


  ///:for dim in [[1,':'],[2,':,:'],[3,':,:,:']]
  ///:for type in types
  subroutine set_ref_farray_${type[0]}$_${dim[0]}$d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_${type[0]}$(A, arr, s) &
            bind(C, name = 'c_set_ref_farray_${type[0]}$')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node) :: A
    ${type[1]}$, target, allocatable, dimension(${dim[1]}$) :: B
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

    call c_set_ref_farray_${type[0]}$(A%ptr, c_loc(B), s3)
  end subroutine
  
  !module procedure set_ref_farray_${type[0]}$_${dim}$
  ///:endfor
  ///:endfor
  

end module oa_set
