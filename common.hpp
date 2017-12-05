#ifndef __COMMON_H__
#define __COMMON_H__

#include <array>

#include "otype.hpp"

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
#define DATA_UNKNOWN -1
#define DATA_INT 0
#define DATA_FLOAT 1
#define DATA_DOUBLE 2
#define DATA_BOOL 3

#define NO_STENCIL 0

// define basic data size
#define DATA_SIZE(i) ((int[]{4, 4, 8})[i])

// define shape dimension [x, y, z]
typedef std::array<int, 3> Shape;
typedef std::array<int, 3> int3;

// void operator= (int* a, const int3& b){
//   a[0] = b[0];
//   a[1] = b[1];
//   a[2] = b[2];
// }

typedef int DataType;
typedef int DATA_TYPE;


///:mute
///:set i = 0  
///:include "NodeType.fypp"
///:endmute
//define node types
enum NodeType{    
  ///:for i in range(len(L))
  ///:if i == 0
  ${L[i][0]}$ = 0,	  
  ///:elif i == len(L) - 1
  ${L[i][0]}$
  ///:else
  ${L[i][0]}$,
  ///:endif    
  ///:endfor
};

#define NUM_NODE_TYPES ${len(L)}$

#include <armadillo>


typedef arma::Cube<int>  cube_int;
typedef arma::Cube<float> cube_float;
typedef arma::Cube<double> cube_double;

#define SCALAR_SHAPE Shape({1,1,1})

#endif

