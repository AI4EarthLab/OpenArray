#ifndef __COMMON_H__
#define __COMMON_H__

#include <array>
#include <assert.h>

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

///:for o in ['+','-','*']
int3 operator${o}$(const int3& a, const int3& b);
///:endfor

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

#define SCALAR_SHAPE Shape({{1,1,1}})

#define WITHOUT_FUSION_KERNEL 0
#define WITH_FUSION_KERNEL 1


#define CSET(A, B)                                    \
  tmp_node_key = gen_node_key(__FILE__, __LINE__);    \
  find_node(tmp_node, tmp_node_key);                  \
  if (is_valid(tmp_node)) {                           \
    A = tmp_node;                                     \
  } else {                                            \
    tmp_node = B;                                     \
    cache_node(tmp_node, tmp_node_key);               \
    A = tmp_node;                                     \
  }

#define THROW_LOGIC_EXCEPTION(msg)              \
  BOOST_THROW_EXCEPTION(std::logic_error(msg))

#define ENSURE_VALID_PTR(A) \
  assert(A && "pointer is null");
  //if(A == NULL) THROW_LOGIC_EXCEPTION("pointer is null.");

#define ENSURE_VALID_PTR_MSG(A, MSG)        \
  assert(A && MSG);
  //if(A == NULL) THROW_LOGIC_EXCEPTION(MSG);

#define MPI_ORDER_START oa::MPI::global()->order_start();
#define MPI_ORDER_END   oa::MPI::global()->order_end();

#define MPI_RANK oa::MPI::global()->rank()
#define MPI_SIZE oa::MPI::global()->size()

extern bool g_cache;
extern bool g_debug;
extern bool transticbegin;

// #define DEBUG

#endif

