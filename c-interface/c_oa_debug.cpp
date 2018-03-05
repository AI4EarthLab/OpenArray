#include "../Function.hpp"
#include "../Operator.hpp"
#include "../Kernel.hpp"
#include "../op_define.hpp"
#include "../MPI.hpp"
#include <cmath>
#include "../Diagnosis.hpp"

extern "C" {
  void c_has_nan_or_inf(int* res, ArrayPtr*& ap, int skip_sw){
    if(oa::diag::has_nan_or_inf(*ap, skip_sw)){
      *res = 1;
    } else{
      *res = 0;
    }
  }
}
