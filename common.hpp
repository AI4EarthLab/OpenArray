#ifndef __COMMON_H__
#define __COMMON_H__

#include <array>

// define random seed
#define SEED 2333333

// define stencil type
#define STENCIL_STAR 0
#define STENCIL_BOX 1

// define boundary type
#define BOUND_OPEN 0
#define BOUNDARY_OPEN 0
#define BOUND_PERIODIC 1
#define BOUNDARY_PERIODIC 1

// define basic data types
#define DATA_INT 0
#define DATA_FLOAT 1
#define DATA_DOUBLE 2
#define DATA_BOOL 3

// define basic data size
#define DATA_SIZE(i) ((int[]{4, 4, 8})[i])
  

// define shape dimension [x, y, z]
typedef std::array<int, 3> Shape;

typedef int DataType;
typedef int DATA_TYPE;


#:mute
#:set i = 0  
#:include "NodeType.fypp"
#:endmute
    //define node types
    enum NodeType{    
#:for i in range(len(L))
#:if i == 0
  ${L[i][0]}$ = 0,	  
#:elif i == len(L) - 1
  ${L[i][0]}$
#:else
      ${L[i][0]}$,
#:endif    
#:endfor
};

#define NUM_NODE_TYPES ${len(L)}$

#include <armadillo>


typedef arma::Cube<int>  cube_int;
typedef arma::Cube<float> cube_float;
typedef arma::Cube<double> cube_double;

/*template<class T>
struct dtype {
  const static DataType type = -1;
  const static int size = -1;
  constexpr static MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
};

template<>
struct dtype<bool>{
  const static DataType type = DATA_BOOL;
  const static int size = 2;
  constexpr static MPI_Datatype mpi_type = MPI_C_BOOL;
};

template<>
struct dtype<int>{
  const static DataType type = DATA_INT;
  const static int size = 4;
  constexpr static MPI_Datatype mpi_type = MPI_INT;
};

template<>
struct dtype<float>{
  const static DataType type = DATA_FLOAT;
  const static int size = 4;
  constexpr static MPI_Datatype mpi_type = MPI_FLOAT;
};

template<>
struct dtype<double>{
  const static DataType type = DATA_DOUBLE;
  const static int size = 8;
  constexpr static MPI_Datatype mpi_type = MPI_DOUBLE;
};*/

template<int N>
struct dtype {
  typedef int type;
  const static int size = -1;
  // const static MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
};

template<>
struct dtype<0> {
  typedef int type;
  const static int size = sizeof(int);
  //const static MPI_Datatype mpi_type = MPI_INT;
};

template<>
struct dtype<1> {
  typedef float type;
  const static int size = sizeof(float);
  //const static MPI_Datatype mpi_type = MPI_FLOAT;
};

template<>
struct dtype<2> {
  typedef double type;
  const static int size = sizeof(double);
  //const static MPI_Datatype mpi_type = MPI_DOUBLE;
};

template<>
struct dtype<3> {
  typedef bool type;
  const static int size = sizeof(bool);
  //const static MPI_Datatype mpi_type = MPI_C_BOOL;
};

#endif

