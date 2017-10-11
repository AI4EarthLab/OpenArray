FC 			= mpif90
CC 			= mpicc
CXX 		= mpicxx
CFLAGS 		=


OBJS 		= Range.o Box.o Partition.o Array.o
OBJS_TEST 	= $(addprefix ./test/, test.o)
OBJS_UTILS	= $(addprefix ./utils/, calcTime.o gettimeofday.o)
OBJ_MAIN 	= ${OBJS} ${OBJS_UTILS} ${OBJS_TEST} main.o

.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) -c $(CFLAGS) $< -o $@

all:
	@rm -rf main
	@echo "Cleaning..."
	@mkdir -p build 2>/dev/null
	@cp makefile build/ 2>/dev/null
	@cp *.cpp *.hpp build/ 2>/dev/null
	@cp -r test build/ 2>/dev/null
	@cp -r utils build/ 2>/dev/null
	@cd build && make clean 
	@echo "Cleaning done."
	@cd build && make main
	cp build/main ./

main: ${OBJ_MAIN}
	-${CXX} -o main ${OBJ_MAIN} -lstdc++ 

small:
	@make all
	@mpirun -n 4 ./main 4 3 2

clean:
	@rm -rf *.o 2>/dev/null


