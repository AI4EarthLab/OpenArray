# OpenArray_CXX
OpenArray is a simple computing library to decouple the work of ocean modelling from the work of parallel computing. The library provides twelve basic operators that feature user-friendly interfaces, effective programming and automatic parallelization.
The main purpose of OpenArray is to make complex parallel programming transparent to the modellers. We use a computation graph as an intermediate representation, meaning that the operator expression forms written in Fortran will be translated into a computation graph with a particular data structure. In addition, OpenArray will use the intermediate computation graph to analyse the dependency of the distributed data and automatically produce the underlying parallel code. Additionally, we use stable and mature compilers, such as the GNU Compiler Collection (GCC), Intel compiler (ICC), and Sunway compiler (SWACC), to generate the executable program according to different backend platforms. These four steps and some related techniques are described in detail in our paper (https://www.geosci-model-dev-discuss.net/gmd-2019-28/).

# Compile OpenArray
Before attempting to compile OpenArray, the following dependent libraries are required:
1.	Fortran 90 or Fortran 95 compiler.
2.	gcc/g++ compiler version 6.1.0 or higher.
3.	Intel icc/icpc compiler version 2017 or higher.
4.	GNU make version 3.81 or higher.
5.	Parallel NetCDF library.
6.	Message Passing Interface (MPI) library.
7.	Armadillo, a C++ library for linear algebra & scientific computing, version 8.200.2 or higher.
8.	Boost C++ Libraries,version 1_65_1 or higher.
9.	LLVM compiler version 6.0.0 or higher.


First checkout to branch dev and type “./test.sh" in the home directory of OpenArray, the source code will be generated in the build folder. Second, change the directory into build, type “make -f makefile.intel oalib_obj”, then libopenarray.a and openarray.mod will be generated if there is no other question.







