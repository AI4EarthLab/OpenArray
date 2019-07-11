
module oa_set
  use iso_c_binding
  use oa_type

  interface set
     module procedure set_ref_const_int
     module procedure set_array_const_int
     module procedure set_ref_const_float
     module procedure set_array_const_float
     module procedure set_ref_const_double
     module procedure set_array_const_double
     module procedure set_ref_array
     module procedure set_ref_ref
     
     module procedure set_array_farray_int_1d
     module procedure set_ref_farray_int_1d

     module procedure set_farray_array_int_1d
     module procedure set_farray_node_int_1d
     module procedure set_array_farray_float_1d
     module procedure set_ref_farray_float_1d

     module procedure set_farray_array_float_1d
     module procedure set_farray_node_float_1d
     module procedure set_array_farray_double_1d
     module procedure set_ref_farray_double_1d

     module procedure set_farray_array_double_1d
     module procedure set_farray_node_double_1d
     module procedure set_array_farray_int_2d
     module procedure set_ref_farray_int_2d

     module procedure set_farray_array_int_2d
     module procedure set_farray_node_int_2d
     module procedure set_array_farray_float_2d
     module procedure set_ref_farray_float_2d

     module procedure set_farray_array_float_2d
     module procedure set_farray_node_float_2d
     module procedure set_array_farray_double_2d
     module procedure set_ref_farray_double_2d

     module procedure set_farray_array_double_2d
     module procedure set_farray_node_double_2d
     module procedure set_array_farray_int_3d
     module procedure set_ref_farray_int_3d

     module procedure set_farray_array_int_3d
     module procedure set_farray_node_int_3d
     module procedure set_array_farray_float_3d
     module procedure set_ref_farray_float_3d

     module procedure set_farray_array_float_3d
     module procedure set_farray_node_float_3d
     module procedure set_array_farray_double_3d
     module procedure set_ref_farray_double_3d

     module procedure set_farray_array_double_3d
     module procedure set_farray_node_double_3d

     !set array/node to a fortran scalar
     module procedure set_int_node
     module procedure set_int_array
     module procedure set_float_node
     module procedure set_float_array
     module procedure set_double_node
     module procedure set_double_array
  end interface set

  interface assignment(=)
     module procedure set_array_const_int
     module procedure set_array_const_float
     module procedure set_array_const_double

     !set fortran array to array object
     module procedure set_array_farray_int_1d
     module procedure set_array_farray_float_1d
     module procedure set_array_farray_double_1d
     module procedure set_array_farray_int_2d
     module procedure set_array_farray_float_2d
     module procedure set_array_farray_double_2d
     module procedure set_array_farray_int_3d
     module procedure set_array_farray_float_3d
     module procedure set_array_farray_double_3d

     !set array/node to a fortran array
     module procedure set_farray_array_int_1d
     module procedure set_farray_node_int_1d
     module procedure set_farray_array_float_1d
     module procedure set_farray_node_float_1d
     module procedure set_farray_array_double_1d
     module procedure set_farray_node_double_1d
     module procedure set_farray_array_int_2d
     module procedure set_farray_node_int_2d
     module procedure set_farray_array_float_2d
     module procedure set_farray_node_float_2d
     module procedure set_farray_array_double_2d
     module procedure set_farray_node_double_2d
     module procedure set_farray_array_int_3d
     module procedure set_farray_node_int_3d
     module procedure set_farray_array_float_3d
     module procedure set_farray_node_float_3d
     module procedure set_farray_array_double_3d
     module procedure set_farray_node_double_3d

     !set array/node to a fortran scalar
     module procedure set_int_node
     module procedure set_int_array
     module procedure set_float_node
     module procedure set_float_array
     module procedure set_double_node
     module procedure set_double_array
  end interface assignment(=)
  
contains

  subroutine set_ref_const_int(A, val)
    implicit none

    interface
       subroutine c_set_ref_const_int(val) &
            bind(C, name="c_set_ref_const_int")
         use iso_c_binding       
         implicit none
         integer, value :: val
       end subroutine
    end interface

    type(node), intent(in) :: A
    integer :: val

    call c_set_ref_const_int(val)

    !call try_destroy(A)
  end subroutine


  subroutine set_array_const_int(A, val)
    implicit none

    interface
       subroutine c_set_array_const_int(A,  val) &
            bind(C, name="c_set_array_const_int")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         integer, value :: val
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, intent(in) :: val
    !print*, "set array const..."
    call c_set_array_const_int(A%ptr, val)

    call try_destroy(A)

  end subroutine
  subroutine set_ref_const_float(A, val)
    implicit none

    interface
       subroutine c_set_ref_const_float(val) &
            bind(C, name="c_set_ref_const_float")
         use iso_c_binding       
         implicit none
         real, value :: val
       end subroutine
    end interface

    type(node), intent(in) :: A
    real :: val

    call c_set_ref_const_float(val)

    !call try_destroy(A)
  end subroutine


  subroutine set_array_const_float(A, val)
    implicit none

    interface
       subroutine c_set_array_const_float(A,  val) &
            bind(C, name="c_set_array_const_float")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real, value :: val
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, intent(in) :: val
    !print*, "set array const..."
    call c_set_array_const_float(A%ptr, val)

    call try_destroy(A)

  end subroutine
  subroutine set_ref_const_double(A, val)
    implicit none

    interface
       subroutine c_set_ref_const_double(val) &
            bind(C, name="c_set_ref_const_double")
         use iso_c_binding       
         implicit none
         real(8), value :: val
       end subroutine
    end interface

    type(node), intent(in) :: A
    real(8) :: val

    call c_set_ref_const_double(val)

    !call try_destroy(A)
  end subroutine


  subroutine set_array_const_double(A, val)
    implicit none

    interface
       subroutine c_set_array_const_double(A,  val) &
            bind(C, name="c_set_array_const_double")
         use iso_c_binding       
         implicit none
         type(c_ptr) :: A
         real(8), value :: val
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), intent(in) :: val
    !print*, "set array const..."
    call c_set_array_const_double(A%ptr, val)

    call try_destroy(A)

  end subroutine

   subroutine set_ref_array(A, B)
     implicit none
     
     interface
        subroutine c_set_ref_array(b) &
             bind(C, name="c_set_ref_array")
          use iso_c_binding
          implicit none
          type(c_ptr) ::  b
        end subroutine
     end interface
     
    type(node), intent(in) :: A
    type(array) :: B
    
    call c_set_ref_array(B%ptr)

    !call try_destroy(A)
    call try_destroy(B)
  end subroutine
  
  subroutine set_ref_ref(A, B)
    implicit none
    interface
        subroutine c_set_ref_ref() &
             bind(C, name="c_set_ref_ref")
          use iso_c_binding
          implicit none
          integer:: id1,id2
        end subroutine
     end interface

    interface
        subroutine c_new_type_set(id1, id2) &
             bind(C, name="c_new_type_set")
          use iso_c_binding
          implicit none
         integer :: id1, id2
        end subroutine
     end interface


    type(node), intent(in) :: A, B
    ! xiaogang
    call c_new_type_set(A%id, B%id)
    call c_set_ref_ref()
    
    !call try_destroy(A)
    !call try_destroy(B)
  end subroutine

  

!cyw modify ======================================================

  !> set_array_farray
  subroutine set_array_farray_int_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_int(A, arr, s) &
            bind(C, name = 'c_set_array_farray_int')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, dimension(:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_array_farray_int(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_float_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_float(A, arr, s) &
            bind(C, name = 'c_set_array_farray_float')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, dimension(:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_array_farray_float(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_double_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_double(A, arr, s) &
            bind(C, name = 'c_set_array_farray_double')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), dimension(:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_array_farray_double(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_int_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_int(A, arr, s) &
            bind(C, name = 'c_set_array_farray_int')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, dimension(:,:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_array_farray_int(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_float_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_float(A, arr, s) &
            bind(C, name = 'c_set_array_farray_float')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, dimension(:,:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_array_farray_float(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_double_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_double(A, arr, s) &
            bind(C, name = 'c_set_array_farray_double')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), dimension(:,:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_array_farray_double(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_int_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_int(A, arr, s) &
            bind(C, name = 'c_set_array_farray_int')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, dimension(:,:,:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_array_farray_int(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_float_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_float(A, arr, s) &
            bind(C, name = 'c_set_array_farray_float')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, dimension(:,:,:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_array_farray_float(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_array_farray_double_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_array_farray_double(A, arr, s) &
            bind(C, name = 'c_set_array_farray_double')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), dimension(:,:,:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_array_farray_double(A%ptr, c_loc(B), s3)

    !call try_destroy(A)

  end subroutine


  !> set_ref_farray
  subroutine set_ref_farray_int_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_int(arr, s) &
            bind(C, name = 'c_set_ref_farray_int')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    integer, dimension(:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_ref_farray_int(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_float_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_float(arr, s) &
            bind(C, name = 'c_set_ref_farray_float')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real, dimension(:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_ref_farray_float(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_double_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_double(arr, s) &
            bind(C, name = 'c_set_ref_farray_double')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real(8), dimension(:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:) :: B
    integer :: s(1), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_ref_farray_double(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_int_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_int(arr, s) &
            bind(C, name = 'c_set_ref_farray_int')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    integer, dimension(:,:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_ref_farray_int(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_float_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_float(arr, s) &
            bind(C, name = 'c_set_ref_farray_float')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real, dimension(:,:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_ref_farray_float(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_double_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_double(arr, s) &
            bind(C, name = 'c_set_ref_farray_double')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real(8), dimension(:,:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:,:) :: B
    integer :: s(2), s3(3)

    s = shape(B)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_ref_farray_double(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_int_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_int(arr, s) &
            bind(C, name = 'c_set_ref_farray_int')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    integer, dimension(:,:,:), target, intent(in) :: B
    
    ! integer, target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_ref_farray_int(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_float_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_float(arr, s) &
            bind(C, name = 'c_set_ref_farray_float')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real, dimension(:,:,:), target, intent(in) :: B
    
    ! real, target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_ref_farray_float(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine
  subroutine set_ref_farray_double_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_ref_farray_double(arr, s) &
            bind(C, name = 'c_set_ref_farray_double')
         use iso_c_binding
         implicit none
         !type(c_ptr),intent(in) :: A !returned array object
         type(c_ptr) :: arr !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    type(node), intent(in):: A
    real(8), dimension(:,:,:), target, intent(in) :: B
    
    ! real(8), target, allocatable, &
    !      dimension(:,:,:) :: B
    integer :: s(3), s3(3)

    s = shape(B)
    s3 = s

    call c_set_ref_farray_double(c_loc(B), s3)

    !call try_destroy(A)

  end subroutine


!cyw modify end



!cyw modify=======================================================

!set_farray_node_
  subroutine set_farray_node_int_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_int(A, s) &
            bind(C, name = 'c_set_farray_node_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_node_int(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_float_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_float(A, s) &
            bind(C, name = 'c_set_farray_node_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_node_float(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_double_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_double(A, s) &
            bind(C, name = 'c_set_farray_node_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_node_double(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_int_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_int(A, s) &
            bind(C, name = 'c_set_farray_node_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_node_int(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_float_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_float(A, s) &
            bind(C, name = 'c_set_farray_node_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_node_float(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_double_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_double(A, s) &
            bind(C, name = 'c_set_farray_node_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_node_double(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_int_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_int(A, s) &
            bind(C, name = 'c_set_farray_node_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:,:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_node_int(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_float_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_float(A, s) &
            bind(C, name = 'c_set_farray_node_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:,:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_node_float(c_loc(A), s3)

  end subroutine

  subroutine set_farray_node_double_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_node_double(A, s) &
            bind(C, name = 'c_set_farray_node_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:,:,:), intent(out) :: A

    type(node), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_node_double(c_loc(A), s3)

  end subroutine


!set_farray array
  subroutine set_farray_array_int_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_int(A, B, s) &
            bind(C, name = 'c_set_farray_array_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_array_int(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_float_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_float(A, B, s) &
            bind(C, name = 'c_set_farray_array_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_array_float(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_double_1d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_double(A, B, s) &
            bind(C, name = 'c_set_farray_array_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(1), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = 1
    s3(3) = 1

    call c_set_farray_array_double(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_int_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_int(A, B, s) &
            bind(C, name = 'c_set_farray_array_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_array_int(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_float_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_float(A, B, s) &
            bind(C, name = 'c_set_farray_array_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_array_float(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_double_2d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_double(A, B, s) &
            bind(C, name = 'c_set_farray_array_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(2), s3(3)

    s = shape(A)
    s3(1) = s(1)
    s3(2) = s(2)
    s3(3) = 1

    call c_set_farray_array_double(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_int_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_int(A, B, s) &
            bind(C, name = 'c_set_farray_array_int')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    integer, target,&
         dimension(:,:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_array_int(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_float_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_float(A, B, s) &
            bind(C, name = 'c_set_farray_array_float')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real, target,&
         dimension(:,:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_array_float(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine

  subroutine set_farray_array_double_3d(A, B)
    use iso_c_binding
    implicit none

    interface
       subroutine c_set_farray_array_double(A, B, s) &
            bind(C, name = 'c_set_farray_array_double')
         use iso_c_binding
         implicit none
         type(c_ptr) :: A !fortran array
         type(c_ptr) :: B !Array object 
         integer :: s(3) !shape
       end subroutine
    end interface

    real(8), target,&
         dimension(:,:,:), intent(out) :: A

    type(array), intent(in) :: B
    
    integer :: s(3), s3(3)

    s = shape(A)
    s3 = s

    call c_set_farray_array_double(c_loc(A), B%ptr, s3)

    call try_destroy(B)
  end subroutine





!cyw modify end

  
 
!cyw modify ========================================================

  subroutine set_int_array(A, B)
    use iso_c_binding
    implicit none
    integer, intent(inout) :: A
    type(array),intent(in) :: B

    interface
       subroutine c_set_int_array(A, B) &
            bind(C, name = "c_set_int_array")
         use iso_c_binding
         implicit none
         type(c_ptr) :: B
         integer(c_int), intent(out) :: A
       end subroutine
    end interface

    call c_set_int_array(A, B%ptr)
    
    call try_destroy(B)
  end subroutine
  subroutine set_float_array(A, B)
    use iso_c_binding
    implicit none
    real, intent(inout) :: A
    type(array),intent(in) :: B

    interface
       subroutine c_set_float_array(A, B) &
            bind(C, name = "c_set_float_array")
         use iso_c_binding
         implicit none
         type(c_ptr) :: B
         real(c_float), intent(out) :: A
       end subroutine
    end interface

    call c_set_float_array(A, B%ptr)
    
    call try_destroy(B)
  end subroutine
  subroutine set_double_array(A, B)
    use iso_c_binding
    implicit none
    real(8), intent(inout) :: A
    type(array),intent(in) :: B

    interface
       subroutine c_set_double_array(A, B) &
            bind(C, name = "c_set_double_array")
         use iso_c_binding
         implicit none
         type(c_ptr) :: B
         real(c_double), intent(out) :: A
       end subroutine
    end interface

    call c_set_double_array(A, B%ptr)
    
    call try_destroy(B)
  end subroutine
 

  subroutine set_int_node(A, B)
    use iso_c_binding
    implicit none
    integer, intent(inout) :: A
    type(node),intent(in) :: B

    interface
       subroutine c_set_int_node(A) &
            bind(C, name = "c_set_int_node")
         use iso_c_binding
         implicit none
         integer(c_int), intent(out) :: A
       end subroutine
    end interface

    call c_set_int_node(A)
    
  !  call try_destroy(B)
  end subroutine
  subroutine set_float_node(A, B)
    use iso_c_binding
    implicit none
    real, intent(inout) :: A
    type(node),intent(in) :: B

    interface
       subroutine c_set_float_node(A) &
            bind(C, name = "c_set_float_node")
         use iso_c_binding
         implicit none
         real(c_float), intent(out) :: A
       end subroutine
    end interface

    call c_set_float_node(A)
    
  !  call try_destroy(B)
  end subroutine
  subroutine set_double_node(A, B)
    use iso_c_binding
    implicit none
    real(8), intent(inout) :: A
    type(node),intent(in) :: B

    interface
       subroutine c_set_double_node(A) &
            bind(C, name = "c_set_double_node")
         use iso_c_binding
         implicit none
         real(c_double), intent(out) :: A
       end subroutine
    end interface

    call c_set_double_node(A)
    
  !  call try_destroy(B)
  end subroutine
 
  
  

  ! transfer distributed fortran array to standard array
  subroutine transfer_dfarray_to_array_int(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
        subroutine c_transfer_dfarray_to_array_int(A, dfa, s3, np, box) &
                bind(C, name = 'c_transfer_dfarray_to_array_int')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(inout) :: A
    integer, dimension(:,:,:), target, intent(in) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_dfarray_to_array_int(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine

  ! transfer standard array to distributed fortran array
  subroutine transfer_array_to_dfarray_int(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
       subroutine c_transfer_array_to_dfarray_int(A, dfa, s3, np, box) &
            bind(C, name = 'c_transfer_dfarray_to_array_int')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(in) :: A
    integer, dimension(:,:,:), target, intent(inout) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_array_to_dfarray_int(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine
  ! transfer distributed fortran array to standard array
  subroutine transfer_dfarray_to_array_float(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
        subroutine c_transfer_dfarray_to_array_float(A, dfa, s3, np, box) &
                bind(C, name = 'c_transfer_dfarray_to_array_float')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real, dimension(:,:,:), target, intent(in) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_dfarray_to_array_float(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine

  ! transfer standard array to distributed fortran array
  subroutine transfer_array_to_dfarray_float(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
       subroutine c_transfer_array_to_dfarray_float(A, dfa, s3, np, box) &
            bind(C, name = 'c_transfer_dfarray_to_array_float')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(in) :: A
    real, dimension(:,:,:), target, intent(inout) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_array_to_dfarray_float(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine
  ! transfer distributed fortran array to standard array
  subroutine transfer_dfarray_to_array_double(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
        subroutine c_transfer_dfarray_to_array_double(A, dfa, s3, np, box) &
                bind(C, name = 'c_transfer_dfarray_to_array_double')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(inout) :: A
    real(8), dimension(:,:,:), target, intent(in) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_dfarray_to_array_double(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine

  ! transfer standard array to distributed fortran array
  subroutine transfer_array_to_dfarray_double(A, dfa, np_x,np_y,np_z,xs,xe,ys,ye,zs,ze)
    use iso_c_binding
    implicit none

    interface
       subroutine c_transfer_array_to_dfarray_double(A, dfa, s3, np, box) &
            bind(C, name = 'c_transfer_dfarray_to_array_double')
         use iso_c_binding
         implicit none
         type(c_ptr),intent(in) :: A
         type(c_ptr) :: dfa
         integer :: s3(3)
         integer :: np(3), box(6)
       end subroutine
    end interface

    type(array), intent(in) :: A
    real(8), dimension(:,:,:), target, intent(inout) :: dfa
    integer, intent(in) :: xs, xe, ys, ye, zs, ze
    integer, intent(in) :: np_x, np_y, np_z

    integer :: s(3), s3(3)
    integer :: np(3), box(6)
    
    np(1) = np_x
    np(2) = np_y
    np(3) = np_z

    box(1) = xs
    box(2) = xe
    box(3) = ys
    box(4) = ye
    box(5) = zs
    box(6) = ze

    s = shape(dfa)
    s3 = s

    !print *, xs,xe,ys,ye,zs,ze
    call c_transfer_array_to_dfarray_double(A%ptr, c_loc(dfa), s3, np, box)

    call try_destroy(A)

  end subroutine


end module oa_set
