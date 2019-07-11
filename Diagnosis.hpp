/*
 * Diagnosis.hpp:
 *
=======================================================*/

#include "MPI.hpp"
#include "ArrayPool.hpp"

namespace oa{
  // for test, check if array has nan or inf value
  namespace diag{
    bool has_nan_or_inf(ArrayPtr& ap, int skip_sw = 1);    
  }
}
