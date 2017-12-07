name = main
compile_thread = -j4
FC = mpif90
CC = mpicc -I${PNETCDF_INC} -L${PNETCDF_LIB} -Werror=return-type \
	  -I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC} -I${GTEST_INC} -L${GTEST}

CXX = mpicxx -O0 -g -fPIC --std=c++0x -Werror=return-type -I${PNETCDF_INC} -L${PNETCDF_LIB} \
		-I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC} -I${GTEST_INC} -L${GTEST} \
		-I${JIT_INC} 


CXXFLAGS= -fPIC --std=c++0x -Werror=return-type -I${PNETCDF_INC} \
	  -L${PNETCDF_LIB} -I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC}

LIBS = -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem \
	-lboost_system -ldl -llapack -lblas \
	-lgtest -L${LAPACK} -L${PNETCDF_LIB} -L${GTEST} libjit.so -lpthread


OBJS	= Range.o Box.o Partition.o Array.o \
	  Internal.o Function.o Kernel.o Operator.o \
	  Node.o IO.o Grid.o

OBJS_UTILS = $(addprefix ./utils/, calcTime.o gettimeofday.o \
			      utils.o)

OBJS_C_INTERFACE = $(addprefix ./c-interface/, c_oa_type.o c_oa_utils.o)

OBJ_FORTRAN = ${OBJS} ${OBJS_UTILS} ${OBJS_C_INTERFACE} \
		$(addprefix ./fortran/, oa_type.o oa_utils.o fortran_main.o)

OBJ_MAIN  = ${OBJS} ${OBJS_UTILS} ${OBJS_C_INTERFACE} main.o

OBJ_TEST = ${OBJS} ${OBJS_UTILS} \
	   $(addprefix ./unittest/, test_array.o gtest_main.o)

OBJ_TEST_PERF = ${OBJS} ${OBJS_UTILS} \
	   $(addprefix ./unittest/, test_perf.o)


.DEFAULT_GOAL := all
Range.o: Range.cpp Range.hpp
Box.o: Box.cpp common.hpp otype.hpp \
 Box.hpp Range.hpp
Partition.o: Partition.cpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp PartitionPool.hpp
Array.o: Array.cpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 utils/../common.hpp ArrayPool.hpp PartitionPool.hpp
Internal.o: Internal.cpp Internal.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 utils/../common.hpp
Function.o: Function.cpp Function.hpp common.hpp otype.hpp \
 utils/../common.hpp Internal.hpp Box.hpp Range.hpp ArrayPool.hpp \
 Array.hpp Partition.hpp PartitionPool.hpp
Kernel.o: Kernel.cpp Kernel.hpp NodePool.hpp common.hpp otype.hpp \
 Node.hpp Array.hpp Partition.hpp Box.hpp Range.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp NodeDesc.hpp
Operator.o: Operator.cpp Operator.hpp NodePool.hpp common.hpp otype.hpp \
 Node.hpp Array.hpp Partition.hpp Box.hpp Range.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp NodeDesc.hpp Kernel.hpp Jit_Driver.hpp \
 Jit.hpp
Node.o: Node.cpp Node.hpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp NodeDesc.hpp Operator.hpp NodePool.hpp ArrayPool.hpp \
 PartitionPool.hpp Function.hpp utils/utils.hpp \
 utils/../common.hpp Internal.hpp
IO.o: IO.cpp IO.hpp Array.hpp Partition.hpp common.hpp otype.hpp \
 Box.hpp Range.hpp utils/utils.hpp \
 Function.hpp Internal.hpp ArrayPool.hpp PartitionPool.hpp
utils.o: utils/utils.cpp utils/utils.hpp \
 utils/../common.hpp utils/../otype.hpp 
c_oa_type.o: c-interface/c_oa_type.cpp c-interface/c_oa_type.hpp \
 c-interface/../ArrayPool.hpp c-interface/../common.hpp \
 c-interface/../otype.hpp \
 c-interface/../Array.hpp c-interface/../Partition.hpp \
 c-interface/../Box.hpp c-interface/../Range.hpp \
 c-interface/../PartitionPool.hpp c-interface/../NodePool.hpp \
 c-interface/../Node.hpp c-interface/../ArrayPool.hpp \
 c-interface/../Function.hpp c-interface/../utils/utils.hpp \
 c-interface/../utils/../common.hpp c-interface/../Internal.hpp \
 c-interface/../Function.hpp c-interface/../Operator.hpp \
 c-interface/../NodePool.hpp c-interface/../NodeDesc.hpp
c_oa_utils.o: c-interface/c_oa_utils.cpp c-interface/c_oa_utils.hpp
main.o: main.cpp test/test.hpp test/../Range.hpp test/../Box.hpp \
 test/../common.hpp test/../otype.hpp \
 test/../Range.hpp test/../Partition.hpp test/../Box.hpp \
 test/../Array.hpp test/../Partition.hpp test/../Function.hpp \
 test/../utils/utils.hpp \
 test/../utils/../common.hpp test/../Internal.hpp test/../ArrayPool.hpp \
 test/../Array.hpp test/../PartitionPool.hpp test/../Internal.hpp \
 test/../Operator.hpp test/../NodePool.hpp test/../Node.hpp \
 test/../Function.hpp test/../NodeDesc.hpp test/../IO.hpp \
 test/../c-interface/c_oa_type.hpp test/../c-interface/../ArrayPool.hpp \
 test/../c-interface/../NodePool.hpp

%.o: %.cpp %.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CXX) -c $(CFLAGS) $< -o $@

%.o: %.F90
	$(FC) -c $< -o $@

all:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./test.sh
	@cd build && make clean
	@echo "Cleaning done."
	@cd build && make main ${compile_thread} 2>/dev/null
	cp build/main ./

quick:
	@rm -rf ${name}
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./test.sh
	@cd build
	@echo "Cleaning done."
	@cd build && make ${name} ${compile_thread}
	@cp build/${name} ./
	@mpirun -n 8 ./${name} 4 4 4

main: ${OBJ_MAIN}
	-${CXX} -rdynamic -o main ${OBJ_MAIN} libjit.so -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system -ldl 

testall:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./test.sh
	@cd build 
	@echo "Cleaning done."
	@cd build && make testall_main ${compile_thread}
	@cp build/testall_main ./
	@mpirun -np 2 ./testall_main 

testall_main : ${OBJ_TEST}
	-${CXX} -o testall_main ${OBJ_TEST} ${LIBS}

testfortran:
	@rm -rf fortran_main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./test.sh
	@cd build && make clean 
	@echo "Cleaning done."
	@cd build && make fortran_main
	@cp build/fortran_main ./
	@mpirun -n 4 ./fortran_main

fortran_main : ${OBJ_FORTRAN}
	-${CXX} -o fortran_main ${OBJ_FORTRAN} -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system \
  
small:
	@make all
	@mpirun -n 4 ./main 4 3 2

clean:
	@rm -rf *.o 2>/dev/null

