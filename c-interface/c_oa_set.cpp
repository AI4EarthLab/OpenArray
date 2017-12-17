#include "c_oa_type.hpp"
#include "../Function.hpp"
#include "../Operator.hpp"

extern "C"{

  ///:for type1 in ['int','float','double']
  void c_set_with_const_${type1}$(void*& A, int* ra, int* rb,int* rc, ${type1}$* val ){
    oa::funcs::set_with_const(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                                 rb[0], rb[1],
                                                 rc[0], rc[1]), 
                                                 *val);
  }
  ///:endfor

  void c_set1(void*& A, int* ra, int* rb,int* rc, void*& B){
    oa::funcs::set(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                      rb[0], rb[1],
                                      rc[0], rc[1]),
                                      *(ArrayPtr*)B);
  }

  void c_set2(void*& A, int* ra, int* rb,int* rc, void*& B, int* sa, int* sb,int* sc){
    oa::funcs::set(*(ArrayPtr*)A, Box(ra[0], ra[1],
                                      rb[0], rb[1],
                                      rc[0], rc[1]),
                                      *(ArrayPtr*)B,
                                  Box(sa[0], sa[1],
                                      sb[0], sb[1],
                                      sc[0], sc[1]));
  }


}
