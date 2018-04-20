/*
 * Init.hpp:
 *  initialize & finalize the OpenArray
 *
=======================================================*/

#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "common.hpp"

namespace oa {

  void init(int comm, Shape procs_shape,
          int argc = 0, char** argv = NULL);

  void finalize();
}

#endif
