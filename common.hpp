#ifndef __COMMON_H__
#define __COMMON_H__

// define stencil type
#define STENCIL_STAR 0
#define STENCIL_BOX 1

// define boundary type
#define BOUND_OPEN 0
#define BOUND_PERIODIC 1

// define basic data types
#define DATA_INT 0
#define DATA_FLOAT 1
#define DATA_DOUBLE 2

// define basic data size
#define DATA_SIZE(i) (int[]{4, 4, 8})[i]

// define shape dimension [x, y, z]
typedef std::array<int, 3> Shape;

#endif
