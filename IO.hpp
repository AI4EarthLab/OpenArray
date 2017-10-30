
#ifndef _IO_HPP__
#define _IO_HPP__

#include "Array.hpp"
#include <string>
#include "mpi.h"

namespace oa{
  namespace io{
    void save(const ArrayPtr& A,
	      const std::string& filename,
	      const std::string& varname);
    
    ArrayPtr load(const std::string& filename,
		  const std::string& varname,
		  const MPI_Comm& comm);
  }
}
#endif
