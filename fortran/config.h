#ifndef __CONFIG_H__
#define __CONFIG_H__

#define STENCIL_WIDTH 1
#define DATA_TYPE 2
#define L 0
#define R 1

///:mute
///:set i = 0  
///:include "../NodeTypeF.fypp"
///:endmute
///:for i in range(len(L))
#define  ${L[i][0]}$  ${i}$
///:endfor


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

#endif
