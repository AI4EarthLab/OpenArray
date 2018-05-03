/*
 * IO.cpp
 * use pnetcdf to support parallel IO
 *
=======================================================*/

#include "IO.hpp"
#include "utils/utils.hpp"
#include "pnetcdf.h"
#include "armadillo"
#include "Function.hpp"
#include "Partition.hpp"

#include <assert.h>

#define CHECK_ERR(status)                               \
  if (status != NC_NOERR) {                             \
    fprintf(stderr, "error found line:%d, msg : %s\n",  \
            __LINE__, ncmpi_strerror(status));          \
    exit(-1);                                           \
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
        err = ncmpi_def_var(ncid, varname.c_str(),
                NC_INT, 3, dimid, &varid);
        CHECK_ERR(err);
        break;
      case(DATA_FLOAT):
        err = ncmpi_def_var(ncid, varname.c_str(),
                NC_FLOAT, 3, dimid, &varid);
        CHECK_ERR(err);
        break;
      case(DATA_DOUBLE):
        err = ncmpi_def_var(ncid, varname.c_str(),
                NC_DOUBLE, 3, dimid, &varid);
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

      bool has_valid_data = count[0] * count[1] * count[2] > 0;
      
      const int bsx = A->buffer_shape()[0];
      const int bsy = A->buffer_shape()[1];
      const int bsz = A->buffer_shape()[2];

      // printf("dim: %d, %d, %d\n", gx, gy, gz);
      // printf("start: %d, %d, %d\n", start[0],start[1],start[2]);
      // printf("count: %d, %d, %d\n", count[0],count[1],count[2]);
      // printf("stride: %d, %d, %d\n",stride[0],stride[1],stride[2]);      

      int sw = A->get_partition()->get_stencil_width();

    
      cube_int c_int;
      cube_float c_float;
      cube_double c_double;
      
      switch(dt) {
        ///:for T in [['INT','int'], ['FLOAT','float'], ['DOUBLE','double']]
      case (DATA_${T[0]}$):
        if(has_valid_data){
          c_${T[1]}$ = utils::make_cube<${T[1]}$>(
              A->buffer_shape(), 
              A->get_buffer())(arma::span(sw, bsx-sw-1),
                      arma::span(sw, bsy-sw-1),
                      arma::span(sw, bsz-sw-1));
          
          err = ncmpi_put_vara_${T[1]}$_all(ncid, varid, start,
                  count, c_${T[1]}$.memptr());
          CHECK_ERR(err);
        }else{
          start[0] = start[1] = start[2] = 0;
          count[0] = count[1] = count[2] = 0;
          err = ncmpi_put_vara_${T[1]}$_all(ncid, varid, start,
                                            count,  NULL);
          CHECK_ERR(err);
        }
        break;
        ///:endfor
      default:
        break;
      }
      ncmpi_close(ncid);
    }

    ArrayPtr load(const std::string& filename, 
                  const std::string& varname,
                  const MPI_Comm& comm) {
      
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

      int sw = Partition::get_default_stencil_width();
      
      ArrayPtr A;

      int xs, ys, zs, xe, ye, ze;
      MPI_Offset starts[3];
      MPI_Offset counts[3];

      cube_int c1_int, c2_int;
      cube_float c1_float, c2_float;
      cube_double c1_double, c2_double;

      void* buf;
      int bsx, bsy, bsz;
      bool is_valid;
      switch(var_type) {
        ///:for T in [['INT','int'], ['FLOAT','float'], ['DOUBLE','double']]
      case(NC_${T[0]}$):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_${T[0]}$);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
  
        starts[0] = zs;
        starts[1] = ys;
        starts[2] = xs;

        counts[0] = ze-zs; 
        counts[1] = ye-ys;
        counts[2] = xe-xs;
      
        is_valid = (counts[0] * counts[1] * counts[2] > 0);

        if(is_valid){
          c1_${T[1]}$ = oa::utils::make_cube<${T[1]}$>(
              A->buffer_shape(),
              A->get_buffer());
        
          c2_${T[1]}$ = oa::utils::make_cube<${T[1]}$>(
              A->local_shape());
        
          status = ncmpi_get_vara_${T[1]}$_all(ncid, varid,
                  starts, counts,
                  c2_${T[1]}$.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_${T[1]}$(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_${T[1]}$;        
        
        }else{
          starts[0] = starts[1] = starts[2] = 0;
          counts[0] = counts[1] = counts[2] = 0;
          status = ncmpi_get_vara_${T[1]}$_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
        ///:endfor
      default:
        break;
      }

      ncmpi_close(ncid);
      return A;

    }
  }
}
