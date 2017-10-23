
#include "IO.hpp"
#include "Utils.hpp"

namespace oa{
  namespace io{
    void save(const ArrayPtr& A,
	      const std::string& filename,
	      const std::string& varname){
      
      oa::utils::mpi_datatype(A.data_type());
      
    };

    ArrayPtr load(const std::string& filename, 
		  const std::string& varname){
      
    };
  }
}
