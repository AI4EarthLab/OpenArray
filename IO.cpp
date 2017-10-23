
#include "IO.hpp"
#include "utils/utils.hpp"

#include "pnetcdf.h"

namespace oa{
  namespace io{
    void save(const ArrayPtr& A,
	      const std::string& filename,
	      const std::string& varname){
      
      DataType dt = A->get_data_type();
      int ncid;
      MPI_Comm comm = A->get_partition()->get_comm();
      Shape arr_shape = A->shape();
      
      int status =
	ncmpi_open(comm, filename.c_str(), NC_WRITE, MPI_INFO_NULL, &ncid);
      assert(status == NC_NOERR);

      int gx = A->shape()[0];
      int gy = A->shape()[1];
      int gz = A->shape()[2];

      int dimid[3];
      ncmpi_def_dim(ncid, "x", gx, &dimid[0]);
      ncmpi_def_dim(ncid, "y", gy, &dimid[1]);
      ncmpi_def_dim(ncid, "z", gz, &dimid[2]);
      
      int varid;

      switch(dt){
      case(DATA_INT):
	ncmpi_def_var(ncid, varname.c_str(), NC_INT, 3, dimid, &varid);
	break;
      case(DATA_FLOAT):
	ncmpi_def_var(ncid, varname.c_str(), NC_FLOAT, 3, dimid, &varid);
	break;
      case(DATA_DOUBLE):
	ncmpi_def_var(ncid, varname.c_str(), NC_DOUBLE, 3, dimid, &varid);
	break;
      }
      ncmpi_enddef(ncid);
      // status = ncmpi_inq_varid(ncid, varname.c_str(), &varid);

      MPI_Offset start[3];
      MPI_Offset count[3];
      MPI_Offset stride[3];

      start[0] = A->get_corners().get_range_x().get_lower();
      start[1] = A->get_corners().get_range_y().get_lower();
      start[2] = A->get_corners().get_range_z().get_lower();
      
      count[0] = A->local_shape()[0];
      count[1] = A->local_shape()[1];
      count[2] = A->local_shape()[2];
      
      stride[0] = A->buffer_shape()[0];
      stride[1] = A->buffer_shape()[1];
      stride[2] = A->buffer_shape()[2];

      switch(dt){
      case(DATA_FLOAT):
	ncmpi_put_vars_float_all(ncid, varid, start,
				 count, stride,
				 (const float*)A->get_buffer());
	break;
      case(DATA_INT):
	ncmpi_put_vars_int_all(ncid, varid, start,
			       count, stride,
			       (const int*)A->get_buffer());
	break;
      case(DATA_DOUBLE):
	ncmpi_put_vars_double_all(ncid, varid, start,
				  count, stride,
				  (const double*)A->get_buffer());
	break;
      }

      ncmpi_close(ncid);
    };

    ArrayPtr load(const std::string& filename, 
		  const std::string& varname){
      
    };
  }
}
