AUTOMAKE_OPTIONS = foreign

oplevel0 = -O0 -w -g  -std=c++0x   -D_WITHOUT_LLVM_ -DSUNWAY
oplevel3 = -O3 -O3 -w -g  -std=c++0x  -D_WITHOUT_LLVM_ -DSUNWAY
FCFLAGS = $(oplevel0)
CFLAGS = $(oplevel0)
CXXFLAGS = $(oplevel0)

if HAS_NPROC
NPROC:=$(shell ${NPROC_PROG})
else
NPROC=2
endif

CC=$(MPICC)
CXX=$(MPICXX)
FC=$(MPIF90)

lib_LIBRARIES = libopenarray.a
libopenarray_a_SOURCES = src/Init.cpp
libopenarray_a_SOURCES += src/Operator.cpp
libopenarray_a_SOURCES += src/ArgumentParser.cpp
libopenarray_a_SOURCES += src/Range.cpp
libopenarray_a_SOURCES += src/MPI.cpp
libopenarray_a_SOURCES += src/log.cpp
libopenarray_a_SOURCES += src/common.cpp
libopenarray_a_SOURCES += src/Array.cpp
libopenarray_a_SOURCES += src/cache.cpp
libopenarray_a_SOURCES += src/Diagnosis.cpp
libopenarray_a_SOURCES += src/IO.cpp
libopenarray_a_SOURCES += src/CUDA.cpp
libopenarray_a_SOURCES += src/Function.cpp
libopenarray_a_SOURCES += src/Box.cpp
libopenarray_a_SOURCES += src/Internal.cpp
libopenarray_a_SOURCES += src/TreeRootDict.cpp
libopenarray_a_SOURCES += src/Node.cpp
libopenarray_a_SOURCES += src/Grid.cpp
libopenarray_a_SOURCES += src/Partition.cpp

libopenarray_a_SOURCES += src/utils/calcTime.cpp
libopenarray_a_SOURCES += src/utils/utils.cpp
libopenarray_a_SOURCES += src/utils/gettimeofday.cpp

libopenarray_a_SOURCES += src/c-interface/c_oa_type.cpp
libopenarray_a_SOURCES += src/c-interface/c_simple_type.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_init.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_utils.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_cache.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_partition.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_grid.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_set_with_mask.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_set.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_debug.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_option.cpp
libopenarray_a_SOURCES += src/c-interface/c_oa_io.cpp

libopenarray_a_SOURCES += src/modules/set/kernel.cpp
libopenarray_a_SOURCES += src/modules/set/new_node.cpp
libopenarray_a_SOURCES += src/modules/tree_tool/NodeVec.cpp
libopenarray_a_SOURCES += src/modules/tree_tool/Simple_Node.cpp
libopenarray_a_SOURCES += src/modules/sum/new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/sum/kernel.cpp
libopenarray_a_SOURCES += src/modules/sum/new_node.cpp
libopenarray_a_SOURCES += src/modules/sum/c_new_node.cpp
libopenarray_a_SOURCES += src/modules/shift/new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/shift/kernel.cpp
libopenarray_a_SOURCES += src/modules/shift/new_node.cpp
libopenarray_a_SOURCES += src/modules/shift/c_new_node.cpp
libopenarray_a_SOURCES += src/modules/interpolation/kernel.cpp
libopenarray_a_SOURCES += src/modules/interpolation/new_node.cpp
libopenarray_a_SOURCES += src/modules/shift/c_new_node.cpp
libopenarray_a_SOURCES += src/modules/interpolation/kernel.cpp
libopenarray_a_SOURCES += src/modules/interpolation/new_node.cpp
libopenarray_a_SOURCES += src/modules/mat_mult/kernel.cpp
libopenarray_a_SOURCES += src/modules/mat_mult/new_node.cpp
libopenarray_a_SOURCES += src/modules/basic/kernel.cpp
libopenarray_a_SOURCES += src/modules/basic/new_node.cpp
libopenarray_a_SOURCES += src/modules/basic/c_new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/basic/c_new_node.cpp
libopenarray_a_SOURCES += src/modules/min_max/new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/min_max/kernel.cpp
libopenarray_a_SOURCES += src/modules/min_max/new_node.cpp
libopenarray_a_SOURCES += src/modules/rep/new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/rep/kernel.cpp
libopenarray_a_SOURCES += src/modules/rep/new_node.cpp
libopenarray_a_SOURCES += src/modules/operator/c_simple_node.cpp
libopenarray_a_SOURCES += src/modules/operator/kernel.cpp
libopenarray_a_SOURCES += src/modules/operator/new_node.cpp
libopenarray_a_SOURCES += src/modules/operator/c_new_node.cpp
libopenarray_a_SOURCES += src/modules/sub/new_node_simple.cpp
libopenarray_a_SOURCES += src/modules/sub/kernel.cpp
libopenarray_a_SOURCES += src/modules/sub/new_node.cpp
libopenarray_a_SOURCES += src/modules/sub/c_new_node.cpp

libopenarray_a_SOURCES += src/fortran/oa_utils.F90
libopenarray_a_SOURCES += src/fortran/oa_type.F90
libopenarray_a_SOURCES += src/fortran/oa_cache.F90
libopenarray_a_SOURCES += src/fortran/oa_init.F90
libopenarray_a_SOURCES += src/fortran/oa_interpolation.F90
libopenarray_a_SOURCES += src/fortran/oa_io.F90
libopenarray_a_SOURCES += src/fortran/oa_mat_mult.F90
libopenarray_a_SOURCES += src/fortran/oa_min_max.F90
libopenarray_a_SOURCES += src/fortran/oa_mpi.F90
libopenarray_a_SOURCES += src/fortran/oa_option.F90
libopenarray_a_SOURCES += src/fortran/oa_partition.F90
libopenarray_a_SOURCES += src/fortran/oa_rep.F90
libopenarray_a_SOURCES += src/fortran/oa_set.F90
libopenarray_a_SOURCES += src/fortran/oa_set_with_mask.F90
libopenarray_a_SOURCES += src/fortran/oa_shift.F90
libopenarray_a_SOURCES += src/fortran/oa_sub.F90
libopenarray_a_SOURCES += src/fortran/oa_sum.F90
libopenarray_a_SOURCES += src/fortran/oa_mod.F90

include_HEADERS = oa_init_fnl.mod
include_HEADERS += oa_min_max.mod
include_HEADERS += oa_option.mod
include_HEADERS += oa_set.mod
include_HEADERS += oa_mat_mult.mod
include_HEADERS += oa_mod.mod
include_HEADERS += oa_io.mod
include_HEADERS += oa_utils.mod
include_HEADERS += oa_sub.mod
include_HEADERS += oa_rep.mod
include_HEADERS += oa_interpolation.mod
include_HEADERS += oa_mpi.mod
include_HEADERS += oa_type.mod
include_HEADERS += oa_set_with_mask.mod
include_HEADERS += oa_sum.mod
include_HEADERS += oa_partition.mod
include_HEADERS += oa_cache.mod
include_HEADERS += oa_shift.mod
include_HEADERS += openarray.mod

# libopenarray_a_SOURCES += src/test/oa_test.F90
# libopenarray_a_SOURCES += src/test/oa_main.F90
libopenarray_a_AR = $(AR) rcs

noinst_PROGRAMS =  manual_main
manual_main_SOURCES = examples/oa_main.F90
manual_main_DEPENDENCIES = libopenarray.a
manual_main_LDADD = -L. -lopenarray -lstdc++ -lgfortran -lm -ldl -lpthread ${AM_LDFLAGS}

examples/oa_main.o: src/fortran/oa_mod.o

oa_init_fnl.mod :   src/fortran/oa_init.F90
src/fortran/oa_cache.o :   src/fortran/oa_cache.F90 oa_utils.mod oa_type.mod
src/fortran/oa_init.o :   src/fortran/oa_init.F90 oa_utils.mod oa_type.mod
src/fortran/oa_interpolation.o: src/fortran/config.h src/fortran/oa_interpolation.F90
src/fortran/oa_io.o :   src/fortran/oa_io.F90 oa_utils.mod oa_type.mod
src/fortran/oa_mat_mult.o: src/fortran/config.h src/fortran/oa_mat_mult.F90
src/fortran/oa_min_max.o: src/fortran/config.h src/fortran/oa_min_max.F90 oa_utils.mod oa_type.mod
src/fortran/oa_mod.o :   src/fortran/oa_mod.F90 oa_interpolation.mod oa_option.mod oa_init_fnl.mod oa_io.mod oa_cache.mod   oa_rep.mod oa_shift.mod oa_min_max.mod oa_sum.mod oa_set_with_mask.mod   oa_set.mod oa_sub.mod oa_partition.mod oa_utils.mod oa_type.mod oa_mpi.mod   oa_interpolation.mod oa_option.mod oa_init_fnl.mod oa_io.mod oa_cache.mod   oa_rep.mod oa_shift.mod oa_min_max.mod oa_sum.mod oa_set_with_mask.mod   oa_set.mod oa_sub.mod oa_partition.mod oa_utils.mod oa_type.mod oa_mpi.mod
src/fortran/oa_mpi.o :   src/fortran/oa_mpi.F90
src/fortran/oa_option.o :   src/fortran/oa_option.F90
src/fortran/oa_partition.o :   src/fortran/oa_partition.F90
src/fortran/oa_rep.o: src/fortran/config.h src/fortran/oa_rep.F90 oa_utils.mod oa_type.mod
src/fortran/oa_set.o :   src/fortran/oa_set.F90 oa_utils.mod oa_type.mod
src/fortran/oa_set_with_mask.o :   src/fortran/oa_set_with_mask.F90 oa_utils.mod oa_type.mod
src/fortran/oa_shift.o: src/fortran/config.h src/fortran/oa_shift.F90 oa_utils.mod oa_type.mod
src/fortran/oa_sub.o :   src/fortran/oa_sub.F90 oa_utils.mod oa_type.mod
src/fortran/oa_sum.o: src/fortran/config.h src/fortran/oa_sum.F90 oa_utils.mod oa_type.mod
src/fortran/oa_type.o: src/fortran/config.h src/fortran/oa_type.F90 oa_utils.mod
src/fortran/oa_utils.o :   src/fortran/oa_utils.F90

oa_init_fnl.mod: src/fortran/oa_init.o

%.mod: src/fortran/%.o
	@true

openarray.mod: libopenarray.a

.PHONY: src bin test

src:
	git archive --format=tar.gz --prefix=${PACKAGE_NAME}-${PACKAGE_VERSION}/ v${PACKAGE_VERSION} > ${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz

bin: libopenarray.a src/openarray.mod
	zip ${PACKAGE_NAME}-bin-release-${PACKAGE_VERSION}.zip $^

test: manual_main
	$(TIME) mpirun -n $(NPROC) ./manual_main

clean-local:
	${RM} -f *.mod

install-data-hook:
	@echo '+----------------------------------------------------------------------------+'
	@echo '|'
	@echo '|  OpenArray has been successfully installed under'
	@echo '|          $(DESTDIR)$(prefix)'
	@echo '|'
	@echo '+----------------------------------------------------------------------------+'
