
module oa_mod
  !use mpi
  use oa_mpi  
  use oa_type
  use oa_partition
  use oa_sub
  use oa_set
  use oa_set_with_mask
  use oa_sum
  use oa_min_max  
  use oa_shift
  use oa_rep
  use oa_cache
  use oa_io
  use oa_init_fnl
  use oa_option
  use oa_utils
  use oa_interpolation
  !use oa_mat_mult
end module

module openarray
  use oa_mod
end module openarray
