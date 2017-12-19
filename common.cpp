
#include "common.hpp"

///:for o in ['+','-','*']
int3 operator${o}$(const int3& a, const int3& b){
  int3 c;
  c[0] = a[0] ${o}$ b[0];
  c[1] = a[1] ${o}$ b[1];
  c[2] = a[2] ${o}$ b[2];
  return c; 
                 }
///:endfor
