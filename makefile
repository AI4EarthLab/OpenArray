FC 			= mpif90
CC 			= mpicc -I${PNETCDF_INC} -L${PNETCDF_LIB}
CXX 		= mpicxx
CFLAGS 		=


OBJS 		= Range.o Box.o Partition.o Array.o \
		  		Internal.o Function.o Operator.o \
		  		Node.o IO.o \


OBJS_UTILS	= $(addprefix ./utils/, calcTime.o gettimeofday.o \
			      utils.o)

OBJ_MAIN 	= ${OBJS} ${OBJS_UTILS} main.o

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
	-${CXX} -o main ${OBJ_MAIN} -lstdc++ -lpnetcdf

small:
	@make all
	@mpirun -n 4 ./main 4 3 2

clean:
	@rm -rf *.o 2>/dev/null


