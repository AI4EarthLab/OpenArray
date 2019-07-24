openarray.mod  oa_mod.mod : \
  fortran/oa_mod.F90

fortran/oa_mod.o : \
  fortran/oa_mod.F90 oa_interpolation.mod oa_option.mod oa_init_fnl.mod oa_io.mod oa_cache.mod \
  oa_rep.mod oa_shift.mod oa_min_max.mod oa_sum.mod oa_set_with_mask.mod \
  oa_set.mod oa_sub.mod oa_partition.mod oa_utils.mod oa_type.mod oa_mpi.mod \
  oa_interpolation.mod oa_option.mod oa_init_fnl.mod oa_io.mod oa_cache.mod \
  oa_rep.mod oa_shift.mod oa_min_max.mod oa_sum.mod oa_set_with_mask.mod \
  oa_set.mod oa_sub.mod oa_partition.mod oa_utils.mod oa_type.mod oa_mpi.mod

