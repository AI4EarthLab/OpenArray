#ifndef __REP_KERNEL_HPP__
#define __REP_KERNEL_HPP__

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"


#include <vector>
using namespace std;

namespace oa{
  namespace kernel{
    ArrayPtr kernel_rep(vector<ArrayPtr> &ops_ap);
    ArrayPtr kernel_rep_with_partition(vector<ArrayPtr> &ops_ap,
            bool same_partition);
  }
}

#endif
