name = main

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
	-lgtest -L${LAPACK} -L${PNETCDF_LIB} -L${GTEST} libjit.so


OBJS	= Range.o Box.o Partition.o Array.o \
	  Internal.o Function.o Kernel.o Operator.o \
	  Node.o IO.o 

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
	@./pre.sh
	@cd build && make clean
	@echo "Cleaning done."
	@cd build && make main  2>/dev/null
	cp build/main ./

quick:
	@rm -rf ${name}
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./test.sh
	@cd build
	@echo "Cleaning done."
	@cd build && make ${name} -j8
	@cp build/${name} ./
	@mpirun -n 8 ./${name} 4 4 4

main: ${OBJ_MAIN}
	-${CXX} -rdynamic -o main ${OBJ_MAIN} libjit.so -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system -ldl 

testall:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./pre.sh
	@cd build && make clean 
	@echo "Cleaning done."
	@cd build && make testall_main
	@cp build/testall_main ./
	@mpirun -np 2 ./testall_main 

testall_main : ${OBJ_TEST}
	-${CXX} -o testall_main ${OBJ_TEST} ${LIBS}

testfortran:
	@rm -rf fortran_main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./pre.sh
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

