#ifndef __OP_DEFINE_HPP__
#define __OP_DEFINE_HPP__

#include "Operator.hpp"
// #include "modules/basic/new_node.hpp"
// #include "modules/operator/new_node.hpp"

///:mute
///:include "NodeType.fypp"
///:endmute
///:for m in MODULES
#include "modules/${m}$/new_node.hpp"
///:endfor

#define NODE(x)           oa::ops::new_node(x)

#define OLD_PLUS(x, y)        oa::ops::new_node(TYPE_PLUS, x, y)
#define OLD_MINUS(x, y)       oa::ops::new_node(TYPE_MINUS, x, y)
#define OLD_MULT(x, y)        oa::ops::new_node(TYPE_MULT, x, y)
#define OLD_DIVD(x, y)        oa::ops::new_node(TYPE_DIVD, x, y)
#define OLD_GT(x, y)          oa::ops::new_node(TYPE_GT, x, y)
#define OLD_GE(x, y)          oa::ops::new_node(TYPE_GE, x, y)
#define OLD_LT(x, y)          oa::ops::new_node(TYPE_LT, x, y)
#define OLD_LE(x, y)          oa::ops::new_node(TYPE_LE, x, y)
#define OLD_EQ(x, y)          oa::ops::new_node(TYPE_EQ, x, y)
#define OLD_NE(x, y)          oa::ops::new_node(TYPE_NE, x, y)
#define OLD_MIN(x, y)         oa::ops::new_node(TYPE_MIN, x, y)
#define OLD_MAX(x, y)         oa::ops::new_node(TYPE_MAX, x, y)
#define OLD_MIN_AT(x, y)      oa::ops::new_node(TYPE_MIN_AT, x, y)
#define OLD_MAX_AT(x, y)      oa::ops::new_node(TYPE_MAX_AT, x, y)
#define OLD_ABS_MIN(x, y)     oa::ops::new_node(TYPE_ABS_MIN, x, y)
#define OLD_ABS_MAX(x, y)     oa::ops::new_node(TYPE_ABS_MAX, x, y)
#define OLD_ABS_MIX_AT(x, y)  oa::ops::new_node(TYPE_ABS_MIN_AT, x, y)
#define OLD_ABS_MAX_AT(x, y)  oa::ops::new_node(TYPE_ABS_MAX_AT, x, y)
#define OLD_POW(x)            oa::ops::new_node(TYPE_POW, x)
#define OLD_EXP(x)            oa::ops::new_node(TYPE_EXP, x)
#define OLD_SIN(x)            oa::ops::new_node(TYPE_SIN, x)
#define OLD_TAN(x)            oa::ops::new_node(TYPE_TAN, x)
#define OLD_COS(x)            oa::ops::new_node(TYPE_COS, x)
#define OLD_RCP(x)            oa::ops::new_node(TYPE_RCP, x)
#define OLD_SQRT(x)           oa::ops::new_node(TYPE_SQRT, x)
#define OLD_ASIN(x)           oa::ops::new_node(TYPE_ASIN, x)
#define OLD_ACOS(x)           oa::ops::new_node(TYPE_ACOS, x)
#define OLD_ATAN(x)           oa::ops::new_node(TYPE_ATAN, x)
#define OLD_ABS(x)            oa::ops::new_node(TYPE_ABS, x)
#define OLD_LOG(x)            oa::ops::new_node(TYPE_LOG, x)
#define OLD_UPLUS(x)          oa::ops::new_node(TYPE_UPLUS, x)
#define OLD_UMINUS(x)         oa::ops::new_node(TYPE_UMINUS, x)
#define OLD_LOG10(x)          oa::ops::new_node(TYPE_LOG10, x)
#define OLD_TANH(x)           oa::ops::new_node(TYPE_TANH, x)
#define OLD_SINH(x)           oa::ops::new_node(TYPE_SINH, x)
#define OLD_COSH(x)           oa::ops::new_node(TYPE_COSH, x)
#define OLD_AXB(x)            oa::ops::new_node(TYPE_AXB, x)
#define OLD_AXF(x)            oa::ops::new_node(TYPE_AXF, x)
#define OLD_AYB(x)            oa::ops::new_node(TYPE_AYB, x)
#define OLD_AYF(x)            oa::ops::new_node(TYPE_AYF, x)
#define OLD_AZB(x)            oa::ops::new_node(TYPE_AZB, x)
#define OLD_AZF(x)            oa::ops::new_node(TYPE_AZF, x)
#define OLD_DXB(x)            oa::ops::new_node(TYPE_DXB, x)
#define OLD_DXF(x)            oa::ops::new_node(TYPE_DXF, x)
#define OLD_DYB(x)            oa::ops::new_node(TYPE_DYB, x)
#define OLD_DYF(x)            oa::ops::new_node(TYPE_DYF, x)
#define OLD_DZB(x)            oa::ops::new_node(TYPE_DZB, x)
#define OLD_DZF(x)            oa::ops::new_node(TYPE_DZF, x)
#define OLD_OR(x)             oa::ops::new_node(TYPE_OR, x)
#define OLD_AND(x)            oa::ops::new_node(TYPE_AND, x)
#define OLD_NOT(x)            oa::ops::new_node(TYPE_NOT, x)

///:include "NodeType.fypp"

///:for t in [i for i in L if i[4] == '2']
///:set name = str.upper(t[1])
#define ${name}$(x, y)  oa::ops::new_node_${t[1]}$(x, y)
///:endfor

///:for t in [i for i in L if i[4] == '1']
///:set name = str.upper(t[1])
#define ${name}$(x)  oa::ops::new_node_${t[1]}$(x)
///:endfor


#define PSU3D(x)          oa::funcs::make_psudo3d(x)
#define EVAL(x)           oa::ops::eval(x)

#endif 
