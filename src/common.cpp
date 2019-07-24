
#include "common.hpp"

oa_int3 operator+(const oa_int3& a, const oa_int3& b){
  oa_int3 c;
  c[0] = a[0] + b[0];
  c[1] = a[1] + b[1];
  c[2] = a[2] + b[2];
  return c; 
                 }
oa_int3 operator-(const oa_int3& a, const oa_int3& b){
  oa_int3 c;
  c[0] = a[0] - b[0];
  c[1] = a[1] - b[1];
  c[2] = a[2] - b[2];
  return c; 
                 }
oa_int3 operator*(const oa_int3& a, const oa_int3& b){
  oa_int3 c;
  c[0] = a[0] * b[0];
  c[1] = a[1] * b[1];
  c[2] = a[2] * b[2];
  return c; 
                 }
