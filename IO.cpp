#include "IO.hpp"
#include "utils/utils.hpp"
#include "pnetcdf.h"
#include "armadillo"
#include <boost/filesystem.hpp>
#include "Function.hpp"

#define CHECK_ERR(status)						\
  if (status != NC_NOERR) {						\
    fprintf(stderr, "error found line:%d, msg : %s\n",			\
	    __LINE__, ncmpi_strerror(status));				\
    exit(-1);								\
  }


namespace oa {
  namespace io {
    void save(const ArrayPtr& A,
        const std::string& filename,
        const std::string& varname) {
      
      DataType dt = A->get_data_type();
      int ncid;
      MPI_Comm comm = A->get_partition()->get_comm();
      Shape arr_shape = A->shape();
      
      int status =
        ncmpi_create(comm, filename.c_str(),
          NC_64BIT_OFFSET, MPI_INFO_NULL, &ncid);
      assert(status == NC_NOERR);

      int gx = A->shape()[0];
      int gy = A->shape()[1];
      int gz = A->shape()[2];

      int dimid[3];
      ncmpi_def_dim(ncid, "x", gx, &dimid[2]);
      ncmpi_def_dim(ncid, "y", gy, &dimid[1]);
      ncmpi_def_dim(ncid, "z", gz, &dimid[0]);
      
      int varid;
      int err;
      
      switch(dt) {
      case(DATA_INT):
        err = ncmpi_def_var(ncid, varname.c_str(), NC_INT, 3, dimid, &varid);
        CHECK_ERR(err);
        break;
      case(DATA_FLOAT):
        err = ncmpi_def_var(ncid, varname.c_str(), NC_FLOAT, 3, dimid, &varid);
        CHECK_ERR(err);
        break;
      case(DATA_DOUBLE):
      	err = ncmpi_def_var(ncid, varname.c_str(), NC_DOUBLE, 3, dimid, &varid);
      	CHECK_ERR(err);
      	break;
      }
      ncmpi_enddef(ncid);

      MPI_Offset start[3], count[3];

      start[2] = A->get_local_box().get_range_x().get_lower();
      start[1] = A->get_local_box().get_range_y().get_lower();
      start[0] = A->get_local_box().get_range_z().get_lower();
      
      count[2] = A->local_shape()[0];
      count[1] = A->local_shape()[1];
      count[0] = A->local_shape()[2];

      const int bsx = A->buffer_shape()[0];
      const int bsy = A->buffer_shape()[1];
      const int bsz = A->buffer_shape()[2];

      // oa::utils::mpi_order_start(MPI_COMM_WORLD);
      // printf("dim: %d, %d, %d\n", gx, gy, gz);
      // printf("start: %d, %d, %d\n", start[0], start[1], start[2]);
      // printf("count: %d, %d, %d\n", count[0], count[1], count[2]);
      // printf("stride: %d, %d, %d\n", stride[0], stride[1], stride[2]);      
      // oa::utils::mpi_order_end(MPI_COMM_WORLD);

      int sw = A->get_partition()->get_stencil_width();

      cube_int c_int;
      cube_float c_float;
      cube_double c_double;
      
      switch(dt) {
        case (DATA_FLOAT):
          c_float = utils::make_cube<float>(A->buffer_shape(), 
                    A->get_buffer())
              (arma::span(sw, bsx-sw-1),
	             arma::span(sw, bsy-sw-1),
	             arma::span(sw, bsz-sw-1));
          
          err = ncmpi_put_vara_float_all(ncid, varid, start,
                    count, 
                    c_float.memptr());
          CHECK_ERR(err);
          break;
        case (DATA_INT):
          c_int = utils::make_cube<int>(A->buffer_shape(),
				          A->get_buffer())
              (arma::span(sw, bsx-sw-1),
               arma::span(sw, bsy-sw-1),
               arma::span(sw, bsz-sw-1));

        	err = ncmpi_put_vara_int_all(ncid, varid, start,
				            count,
				            c_int.memptr());
	        CHECK_ERR(err);
          break;
        case (DATA_DOUBLE):
	        c_double = utils::make_cube<double>(A->buffer_shape(),
				   	         A->get_buffer());
	            (arma::span(sw, bsx-sw-1),
	             arma::span(sw, bsy-sw-1),
	             arma::span(sw, bsz-sw-1));
   
	        err = ncmpi_put_vara_double_all(ncid, varid, start,
				   	        count,
				   	        c_double.memptr());
          CHECK_ERR(err);
	        break;
        default:
	        break;
      }
      ncmpi_close(ncid);
    }

    ArrayPtr load(const std::string& filename, 
		  const std::string& varname,
		  const MPI_Comm& comm) {
      
      //assert(boost::filesystem::exists(filename.c_str()));

      int status;
      int ncid, varid;
      nc_type var_type;
      int var_dimids[3];
      int ndim;
      
      status = ncmpi_open(comm, filename.c_str(),
			  NC_NOWRITE, MPI_INFO_NULL, &ncid);

      CHECK_ERR(status);
      
      status = ncmpi_inq_varid(ncid, varname.c_str(), &varid);

      CHECK_ERR(status);

      status = ncmpi_inq_var(ncid, varid, 0, &var_type, &ndim,
			     var_dimids, NULL);
      CHECK_ERR(status);

      MPI_Offset gx, gy, gz;
      status = ncmpi_inq_dimlen(ncid, var_dimids[0], &gz);
      CHECK_ERR(status);

      status = ncmpi_inq_dimlen(ncid, var_dimids[1], &gy);
      CHECK_ERR(status);

      status = ncmpi_inq_dimlen(ncid, var_dimids[2], &gx);
      CHECK_ERR(status);

      int sw = 1;
      
      ArrayPtr A;

      int xs, ys, zs, xe, ye, ze;
      MPI_Offset starts[3];
      MPI_Offset counts[3];

      cube_int c1_int, c2_int;
      cube_float c1_float, c2_float;
      cube_double c1_double, c2_double;

      void* buf;
      int bsx, bsy, bsz;
      
      switch(var_type) {
#:for T in [['NC_INT', 'DATA_INT', 'int'], &
	['NC_FLOAT', 'DATA_FLOAT', 'float'],	&
	  ['NC_DOUBLE', 'DATA_DOUBLE', 'double']] 	  
      case ${T[0]}$:
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, ${T[1]}$);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
	
        starts[0] = zs;
        starts[1] = ys;
        starts[2] = xs;

        counts[0] = ze-zs; 
        counts[1] = ye-ys;
        counts[2] = xe-xs;

        c1_${T[2]}$ = oa::utils::make_cube<${T[2]}$>(A->buffer_shape(),
        				A->get_buffer());
        
        c2_${T[2]}$ = oa::utils::make_cube<${T[2]}$>(A->local_shape());
        
        status = ncmpi_get_vara_${T[2]}$_all(ncid, varid,
        				starts, counts,
        				c2_${T[2]}$.memptr());

        bsx = A->buffer_shape()[0];
        bsy = A->buffer_shape()[1];
        bsz = A->buffer_shape()[2];
	
        c1_${T[2]}$(arma::span(sw, bsx-sw-1),
            arma::span(sw, bsy-sw-1),
            arma::span(sw, bsz-sw-1)) = c2_${T[2]}$;        
        
        CHECK_ERR(status);
        break;
#:endfor
      default:
        break;
      }

      ncmpi_close(ncid);
				     //A->display("====AAAA====");
      return A;
    }

  }
}
