#ifndef __COMMON_H__
#define __COMMON_H__

#include <array>
#include <assert.h>

#include "otype.hpp"

//xiaogang

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
typedef std::array<int, 3> oa_int3;

oa_int3 operator+(const oa_int3& a, const oa_int3& b);
oa_int3 operator-(const oa_int3& a, const oa_int3& b);
oa_int3 operator*(const oa_int3& a, const oa_int3& b);

typedef int DataType;
typedef int DATA_TYPE;


//define node types
enum NodeType{    
  TYPE_UNKNOWN = 0,	  
  TYPE_DATA,
  TYPE_REF,
  TYPE_PLUS,
  TYPE_MINUS,
  TYPE_MULT,
  TYPE_DIVD,
  TYPE_GT,
  TYPE_GE,
  TYPE_LT,
  TYPE_LE,
  TYPE_EQ,
  TYPE_NE,
  TYPE_MIN,
  TYPE_MAX,
  TYPE_MIN_AT,
  TYPE_MAX_AT,
  TYPE_ABS_MAX,
  TYPE_ABS_MIN,
  TYPE_ABS_MAX_AT,
  TYPE_ABS_MIN_AT,
  TYPE_MIN2,
  TYPE_MAX2,
  TYPE_POW,
  TYPE_EXP,
  TYPE_SIN,
  TYPE_TAN,
  TYPE_COS,
  TYPE_RCP,
  TYPE_SQRT,
  TYPE_ASIN,
  TYPE_ACOS,
  TYPE_ATAN,
  TYPE_ABS,
  TYPE_LOG,
  TYPE_UPLUS,
  TYPE_UMINUS,
  TYPE_LOG10,
  TYPE_TANH,
  TYPE_SINH,
  TYPE_COSH,
  TYPE_DXC,
  TYPE_DYC,
  TYPE_DZC,
  TYPE_AXB,
  TYPE_AXF,
  TYPE_AYB,
  TYPE_AYF,
  TYPE_AZB,
  TYPE_AZF,
  TYPE_DXB,
  TYPE_DXF,
  TYPE_DYB,
  TYPE_DYF,
  TYPE_DZB,
  TYPE_DZF,
  TYPE_SUM,
  TYPE_CSUM,
  TYPE_OR,
  TYPE_AND,
  TYPE_NOT,
  TYPE_REP,
  TYPE_SHIFT,
  TYPE_CIRCSHIFT,
  TYPE_INT,
  TYPE_FLOAT,
  TYPE_DOUBLE,
  TYPE_INT3_REP,
  TYPE_INT3_SHIFT,
  TYPE_SET
};

enum DeviceType {DEVICE_NONE=0, CPU, GPU, CPU_AND_GPU};

#define NUM_NODE_TYPES 70

#include "armadillo"


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
extern int g_flag;
extern bool g_debug;
extern bool transticbegin;

// #define DEBUG

#endif

