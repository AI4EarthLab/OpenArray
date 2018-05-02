/*
 * IO.hpp
 *
=======================================================*/

#ifndef _IO_HPP__
#define _IO_HPP__

#include "Array.hpp"
#include <string>
#include "mpi.h"

namespace oa {
  namespace io {
    // save array varname into filename
    void save (const ArrayPtr& A,
        const std::string& filename,
        const std::string& varname);
    
    // load varname from filename
    ArrayPtr load (const std::string& filename,
      const std::string& varname,
      const MPI_Comm& comm);
  }
}
#endif
