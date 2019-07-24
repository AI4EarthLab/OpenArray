#ifndef __OP_DEFINE_HPP__
#define __OP_DEFINE_HPP__

#include "Operator.hpp"
// #include "modules/basic/new_node.hpp"
// #include "modules/operator/new_node.hpp"

#include "modules/mat_mult/new_node.hpp"
#include "modules/min_max/new_node.hpp"
#include "modules/sum/new_node.hpp"
#include "modules/interpolation/new_node.hpp"
#include "modules/set/new_node.hpp"
#include "modules/tree_tool/new_node.hpp"
#include "modules/sub/new_node.hpp"
#include "modules/basic/new_node.hpp"
#include "modules/shift/new_node.hpp"
#include "modules/rep/new_node.hpp"
#include "modules/operator/new_node.hpp"

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

  
  






#define PLUS(x, y)  oa::ops::new_node_plus(x, y)
#define MINUS(x, y)  oa::ops::new_node_minus(x, y)
#define MULT(x, y)  oa::ops::new_node_mult(x, y)
#define DIVD(x, y)  oa::ops::new_node_divd(x, y)
#define GT(x, y)  oa::ops::new_node_gt(x, y)
#define GE(x, y)  oa::ops::new_node_ge(x, y)
#define LT(x, y)  oa::ops::new_node_lt(x, y)
#define LE(x, y)  oa::ops::new_node_le(x, y)
#define EQ(x, y)  oa::ops::new_node_eq(x, y)
#define NE(x, y)  oa::ops::new_node_ne(x, y)
#define POW(x, y)  oa::ops::new_node_pow(x, y)
#define SUM(x, y)  oa::ops::new_node_sum(x, y)
#define CSUM(x, y)  oa::ops::new_node_csum(x, y)
#define OR(x, y)  oa::ops::new_node_or(x, y)
#define AND(x, y)  oa::ops::new_node_and(x, y)
#define SHIFT(x, y)  oa::ops::new_node_shift(x, y)
#define CIRCSHIFT(x, y)  oa::ops::new_node_circshift(x, y)
#define SET(x, y)  oa::ops::new_node_set(x, y)

#define MIN(x)  oa::ops::new_node_min(x)
#define MAX(x)  oa::ops::new_node_max(x)
#define MIN_AT(x)  oa::ops::new_node_min_at(x)
#define MAX_AT(x)  oa::ops::new_node_max_at(x)
#define ABS_MAX(x)  oa::ops::new_node_abs_max(x)
#define ABS_MIN(x)  oa::ops::new_node_abs_min(x)
#define ABS_MAX_AT(x)  oa::ops::new_node_abs_max_at(x)
#define ABS_MIN_AT(x)  oa::ops::new_node_abs_min_at(x)
#define EXP(x)  oa::ops::new_node_exp(x)
#define SIN(x)  oa::ops::new_node_sin(x)
#define TAN(x)  oa::ops::new_node_tan(x)
#define COS(x)  oa::ops::new_node_cos(x)
#define RCP(x)  oa::ops::new_node_rcp(x)
#define SQRT(x)  oa::ops::new_node_sqrt(x)
#define ASIN(x)  oa::ops::new_node_asin(x)
#define ACOS(x)  oa::ops::new_node_acos(x)
#define ATAN(x)  oa::ops::new_node_atan(x)
#define ABS(x)  oa::ops::new_node_abs(x)
#define LOG(x)  oa::ops::new_node_log(x)
#define UPLUS(x)  oa::ops::new_node_uplus(x)
#define UMINUS(x)  oa::ops::new_node_uminus(x)
#define LOG10(x)  oa::ops::new_node_log10(x)
#define TANH(x)  oa::ops::new_node_tanh(x)
#define SINH(x)  oa::ops::new_node_sinh(x)
#define COSH(x)  oa::ops::new_node_cosh(x)
#define DXC(x)  oa::ops::new_node_dxc(x)
#define DYC(x)  oa::ops::new_node_dyc(x)
#define DZC(x)  oa::ops::new_node_dzc(x)
#define AXB(x)  oa::ops::new_node_axb(x)
#define AXF(x)  oa::ops::new_node_axf(x)
#define AYB(x)  oa::ops::new_node_ayb(x)
#define AYF(x)  oa::ops::new_node_ayf(x)
#define AZB(x)  oa::ops::new_node_azb(x)
#define AZF(x)  oa::ops::new_node_azf(x)
#define DXB(x)  oa::ops::new_node_dxb(x)
#define DXF(x)  oa::ops::new_node_dxf(x)
#define DYB(x)  oa::ops::new_node_dyb(x)
#define DYF(x)  oa::ops::new_node_dyf(x)
#define DZB(x)  oa::ops::new_node_dzb(x)
#define DZF(x)  oa::ops::new_node_dzf(x)
#define NOT(x)  oa::ops::new_node_not(x)


#define PSU3D(x)          oa::funcs::make_psudo3d(x)
#define EVAL(x)           oa::ops::eval(x)

#endif 
