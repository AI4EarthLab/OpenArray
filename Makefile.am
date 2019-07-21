AUTOMAKE_OPTIONS = foreign

oplevel0 = -O0 -w -g  -std=c++0x   -D_WITHOUT_LLVM_ -DSUNWAY
oplevel3 = -O3 -O3 -w -g  -std=c++0x  -D_WITHOUT_LLVM_ -DSUNWAY
FCFLAGS = $(oplevel0)
CFLAGS = $(oplevel0)
CXXFLAGS = $(oplevel0)

lib_LIBRARIES = libopenarray.a
libopenarray_a_SOURCES = ./Init.cpp
libopenarray_a_SOURCES += ./Operator.cpp
libopenarray_a_SOURCES += ./ArgumentParser.cpp
libopenarray_a_SOURCES += ./Range.cpp
libopenarray_a_SOURCES += ./MPI.cpp
libopenarray_a_SOURCES += ./log.cpp
libopenarray_a_SOURCES += ./common.cpp
libopenarray_a_SOURCES += ./Array.cpp
libopenarray_a_SOURCES += ./cache.cpp
libopenarray_a_SOURCES += ./Diagnosis.cpp
libopenarray_a_SOURCES += ./IO.cpp
libopenarray_a_SOURCES += ./CUDA.cpp
libopenarray_a_SOURCES += ./Function.cpp
libopenarray_a_SOURCES += ./Box.cpp
libopenarray_a_SOURCES += ./Internal.cpp
libopenarray_a_SOURCES += ./TreeRootDict.cpp
libopenarray_a_SOURCES += ./Node.cpp
libopenarray_a_SOURCES += ./Grid.cpp
libopenarray_a_SOURCES += ./Partition.cpp

libopenarray_a_SOURCES += ./utils/calcTime.cpp
libopenarray_a_SOURCES += ./utils/utils.cpp
libopenarray_a_SOURCES += ./utils/gettimeofday.cpp

libopenarray_a_SOURCES += ./c-interface/c_oa_type.cpp
libopenarray_a_SOURCES += ./c-interface/c_simple_type.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_init.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_utils.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_cache.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_partition.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_grid.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_set_with_mask.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_set.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_debug.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_option.cpp
libopenarray_a_SOURCES += ./c-interface/c_oa_io.cpp

libopenarray_a_SOURCES += modules/set/kernel.cpp
libopenarray_a_SOURCES += modules/set/new_node.cpp
libopenarray_a_SOURCES += modules/tree_tool/NodeVec.cpp
libopenarray_a_SOURCES += modules/tree_tool/Simple_Node.cpp
libopenarray_a_SOURCES += modules/sum/new_node_simple.cpp
libopenarray_a_SOURCES += modules/sum/kernel.cpp
libopenarray_a_SOURCES += modules/sum/new_node.cpp
libopenarray_a_SOURCES += modules/sum/c_new_node.cpp
libopenarray_a_SOURCES += modules/shift/new_node_simple.cpp
libopenarray_a_SOURCES += modules/shift/kernel.cpp
libopenarray_a_SOURCES += modules/shift/new_node.cpp
libopenarray_a_SOURCES += modules/shift/c_new_node.cpp
libopenarray_a_SOURCES += modules/interpolation/kernel.cpp
libopenarray_a_SOURCES += modules/interpolation/new_node.cpp
libopenarray_a_SOURCES += modules/shift/c_new_node.cpp
libopenarray_a_SOURCES += modules/interpolation/kernel.cpp
libopenarray_a_SOURCES += modules/interpolation/new_node.cpp
libopenarray_a_SOURCES += modules/mat_mult/kernel.cpp
libopenarray_a_SOURCES += modules/mat_mult/new_node.cpp
libopenarray_a_SOURCES += modules/basic/kernel.cpp
libopenarray_a_SOURCES += modules/basic/new_node.cpp
libopenarray_a_SOURCES += modules/basic/c_new_node_simple.cpp
libopenarray_a_SOURCES += modules/basic/c_new_node.cpp
libopenarray_a_SOURCES += modules/min_max/new_node_simple.cpp
libopenarray_a_SOURCES += modules/min_max/kernel.cpp
libopenarray_a_SOURCES += modules/min_max/new_node.cpp
libopenarray_a_SOURCES += modules/rep/new_node_simple.cpp
libopenarray_a_SOURCES += modules/rep/kernel.cpp
libopenarray_a_SOURCES += modules/rep/new_node.cpp
libopenarray_a_SOURCES += modules/operator/c_simple_node.cpp
libopenarray_a_SOURCES += modules/operator/kernel.cpp
libopenarray_a_SOURCES += modules/operator/new_node.cpp
libopenarray_a_SOURCES += modules/operator/c_new_node.cpp
libopenarray_a_SOURCES += modules/sub/new_node_simple.cpp
libopenarray_a_SOURCES += modules/sub/kernel.cpp
libopenarray_a_SOURCES += modules/sub/new_node.cpp
libopenarray_a_SOURCES += modules/sub/c_new_node.cpp

libopenarray_a_SOURCES += fortran/oa_utils.F90
libopenarray_a_SOURCES += fortran/oa_type.F90
libopenarray_a_SOURCES += fortran/oa_cache.F90
libopenarray_a_SOURCES += fortran/oa_init.F90
libopenarray_a_SOURCES += fortran/oa_interpolation.F90
libopenarray_a_SOURCES += fortran/oa_io.F90
libopenarray_a_SOURCES += fortran/oa_mat_mult.F90
libopenarray_a_SOURCES += fortran/oa_min_max.F90
libopenarray_a_SOURCES += fortran/oa_mpi.F90
libopenarray_a_SOURCES += fortran/oa_option.F90
libopenarray_a_SOURCES += fortran/oa_partition.F90
libopenarray_a_SOURCES += fortran/oa_rep.F90
libopenarray_a_SOURCES += fortran/oa_set.F90
libopenarray_a_SOURCES += fortran/oa_set_with_mask.F90
libopenarray_a_SOURCES += fortran/oa_shift.F90
libopenarray_a_SOURCES += fortran/oa_sub.F90
libopenarray_a_SOURCES += fortran/oa_sum.F90
libopenarray_a_SOURCES += fortran/oa_mod.F90

include_HEADERS = openarray.mod

# libopenarray_a_SOURCES += test/oa_test.F90
# libopenarray_a_SOURCES += test/oa_main.F90
libopenarray_a_AR = $(AR) rcs

noinst_PROGRAMS =  manual_main
manual_main_SOURCES = user-manual/oa_main.F90
manual_main_DEPENDENCIES = libopenarray.a
manual_main_LDADD = -L. -lopenarray -lstdc++ -lgfortran -lm -ldl -lpthread ${AM_LDFLAGS}

user-manual/oa_main.o: fortran/oa_mod.o

fortran/oa_type.o: fortran/oa_utils.o
fortran/oa_cache.o: fortran/oa_type.o
fortran/oa_io.o: fortran/oa_type.o
fortran/oa_init.o: fortran/oa_type.o
fortran/oa_min_max.o: fortran/oa_type.o
fortran/oa_rep.o: fortran/oa_type.o
fortran/oa_sum.o: fortran/oa_type.o
fortran/oa_set_with_mask.o: fortran/oa_type.o
fortran/oa_shift.o: fortran/oa_type.o
fortran/oa_set.o: fortran/oa_type.o
fortran/oa_sub.o: fortran/oa_type.o
fortran/oa_mod.o: fortran/oa_mpi.o fortran/oa_type.o fortran/oa_sub.o fortran/oa_partition.o
openarray.mod: libopenarray.a

.PHONY: src bin

src:
	git archive --format=tar.gz --prefix=${PACKAGE_NAME}-${PACKAGE_VERSION}/ v${PACKAGE_VERSION} > ${PACKAGE_NAME}-${PACKAGE_VERSION}.tar.gz

bin: libopenarray.a openarray.mod
	zip ${PACKAGE_NAME}-bin-release-${PACKAGE_VERSION}.zip $^
