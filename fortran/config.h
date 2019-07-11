#ifndef __CONFIG_H__
#define __CONFIG_H__

#define L 0
#define R 1

#define LVALUE 0
#define RVALUE 1

#define  TYPE_UNKNOWN  0
#define  TYPE_DATA  1
#define  TYPE_REF  2
#define  TYPE_PLUS  3
#define  TYPE_MINUS  4
#define  TYPE_MULT  5
#define  TYPE_DIVD  6
#define  TYPE_GT  7
#define  TYPE_GE  8
#define  TYPE_LT  9
#define  TYPE_LE  10
#define  TYPE_EQ  11
#define  TYPE_NE  12
#define  TYPE_MIN  13
#define  TYPE_MAX  14
#define  TYPE_MIN_AT  15
#define  TYPE_MAX_AT  16
#define  TYPE_ABS_MAX  17
#define  TYPE_ABS_MIN  18
#define  TYPE_ABS_MAX_AT  19
#define  TYPE_ABS_MIN_AT  20
#define  TYPE_MIN2  21
#define  TYPE_MAX2  22
#define  TYPE_POW  23
#define  TYPE_EXP  24
#define  TYPE_SIN  25
#define  TYPE_TAN  26
#define  TYPE_COS  27
#define  TYPE_RCP  28
#define  TYPE_SQRT  29
#define  TYPE_ASIN  30
#define  TYPE_ACOS  31
#define  TYPE_ATAN  32
#define  TYPE_ABS  33
#define  TYPE_LOG  34
#define  TYPE_UPLUS  35
#define  TYPE_UMINUS  36
#define  TYPE_LOG10  37
#define  TYPE_TANH  38
#define  TYPE_SINH  39
#define  TYPE_COSH  40
#define  TYPE_DXC  41
#define  TYPE_DYC  42
#define  TYPE_DZC  43
#define  TYPE_AXB  44
#define  TYPE_AXF  45
#define  TYPE_AYB  46
#define  TYPE_AYF  47
#define  TYPE_AZB  48
#define  TYPE_AZF  49
#define  TYPE_DXB  50
#define  TYPE_DXF  51
#define  TYPE_DYB  52
#define  TYPE_DYF  53
#define  TYPE_DZB  54
#define  TYPE_DZF  55
#define  TYPE_SUM  56
#define  TYPE_CSUM  57
#define  TYPE_OR  58
#define  TYPE_AND  59
#define  TYPE_NOT  60
#define  TYPE_REP  61
#define  TYPE_SHIFT  62
#define  TYPE_CIRCSHIFT  63
#define  TYPE_INT  64
#define  TYPE_FLOAT  65
#define  TYPE_DOUBLE  66
#define  TYPE_INT3_REF  67
#define  TYPE_INT3_SHIFT  68
#define  TYPE_SET  69


#define FSET(A, B)                              \
  call gen_node_key__(__FILE__,__LINE__);       \
  call find_node__();                           \
  if(is_valid__()) then;                        \
  A = tmp_node__;                               \
  else;                                         \
  tmp_node__ = B;                               \
  call cache_node__();                          \
  A = tmp_node__;                               \
  end if;


#define ASSERT(cond, msg)                       \
  call assert(cond, __FILE__, __LINE__, msg);

#define ASSERT_LVALUE(A)                                        \
   ASSERT(.not. is_rvalue(A), "object must be lvalue.");

#endif
