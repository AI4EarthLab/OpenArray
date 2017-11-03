
FC = mpif90
CC = mpicc -I${PNETCDF_INC} -L${PNETCDF_LIB} \
	  -I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC}


CXX = mpicxx --std=c++0x -I${PNETCDF_INC} -L${PNETCDF_LIB} \
		-I${ARMA_INC} -I${ARMA_LIB} -I${BOOST_INC}

OBJS 		= Range.o Box.o Partition.o Array.o \
		  Internal.o Function.o Kernel.o Operator.o \
		  Node.o IO.o \

CFLAGS	=


OBJS_UTILS = $(addprefix ./utils/, calcTime.o gettimeofday.o \
			      utils.o)

OBJ_MAIN  = ${OBJS} ${OBJS_UTILS} main.o

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
	-${CXX} -rdynamic -o main ${OBJ_MAIN} -lstdc++ -lpnetcdf \
	-lboost_program_options -lboost_filesystem -lboost_system -ldl

small:
	@make all
	@mpirun -n 4 ./main 4 3 2

clean:
	@rm -rf *.o 2>/dev/null


