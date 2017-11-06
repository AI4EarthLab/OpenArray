
FC = mpif90
CC = mpicc -I${PNETCDF_INC} -L${PNETCDF_LIB} -Werror=return-type \
	  -I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC}


CXX = mpicxx --std=c++0x -Werror=return-type -I${PNETCDF_INC} -L${PNETCDF_LIB} \
		-I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC}

OBJS 		= Range.o Box.o Partition.o Array.o \
		  Internal.o Function.o Kernel.o Operator.o \
		  Node.o IO.o \

CFLAGS	=


OBJS_UTILS = $(addprefix ./utils/, calcTime.o gettimeofday.o \
			      utils.o)

OBJ_MAIN  = ${OBJS} ${OBJS_UTILS} main.o

OBJ_TEST = ${OBJS} ${OBJS_UTILS} \
	   $(addprefix ./unittest/, test_array.o gtest_main.o)


.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

all:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./pre.sh
	@cd build && make clean 
	@echo "Cleaning done."
	@cd build && make main
	cp build/main ./

main: ${OBJ_MAIN}
	-${CXX} -o main ${OBJ_MAIN} -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system

testall:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@./pre.sh
	@cd build && make clean 
	@echo "Cleaning done."
	@cd build && make test_main
	@cp build/test_main ./
	@mpirun -n 2 ./test_main 

test_main : ${OBJ_TEST}
	-${CXX} -o test_main ${OBJ_TEST} -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system \
        -lgtest 

small:
	@make all
	@mpirun -n 4 ./main 4 3 2

clean:
	@rm -rf *.o 2>/dev/null


