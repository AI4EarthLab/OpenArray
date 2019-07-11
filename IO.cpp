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

#define VAR_1D 1
#define VAR_2D 2
#define VAR_3D 3
#define VAR_4D 4
#define VAR_5D 5


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

      #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
      #endif  

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
      case (DATA_INT):
        if(has_valid_data){
          c_int = utils::make_cube<int>(
              A->buffer_shape(), 
              A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                      arma::span(sw, bsy-sw-1),
                      arma::span(sw, bsz-sw-1));
          
          err = ncmpi_put_vara_int_all(ncid, varid, start,
                  count, c_int.memptr());
          CHECK_ERR(err);
        }else{
          start[0] = start[1] = start[2] = 0;
          count[0] = count[1] = count[2] = 0;
          err = ncmpi_put_vara_int_all(ncid, varid, start,
                                            count,  NULL);
          CHECK_ERR(err);
        }
        break;
      case (DATA_FLOAT):
        if(has_valid_data){
          c_float = utils::make_cube<float>(
              A->buffer_shape(), 
              A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                      arma::span(sw, bsy-sw-1),
                      arma::span(sw, bsz-sw-1));
          
          err = ncmpi_put_vara_float_all(ncid, varid, start,
                  count, c_float.memptr());
          CHECK_ERR(err);
        }else{
          start[0] = start[1] = start[2] = 0;
          count[0] = count[1] = count[2] = 0;
          err = ncmpi_put_vara_float_all(ncid, varid, start,
                                            count,  NULL);
          CHECK_ERR(err);
        }
        break;
      case (DATA_DOUBLE):
        if(has_valid_data){
          c_double = utils::make_cube<double>(
              A->buffer_shape(), 
              A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                      arma::span(sw, bsy-sw-1),
                      arma::span(sw, bsz-sw-1));
          
          err = ncmpi_put_vara_double_all(ncid, varid, start,
                  count, c_double.memptr());
          CHECK_ERR(err);
        }else{
          start[0] = start[1] = start[2] = 0;
          count[0] = count[1] = count[2] = 0;
          err = ncmpi_put_vara_double_all(ncid, varid, start,
                                            count,  NULL);
          CHECK_ERR(err);
        }
        break;
      default:
        break;
      }

      #ifdef __HAVE_CUDA__
      A->set_newest_buffer(GPU);
      #endif
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
      case(NC_INT):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_INT);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);

        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif 
        starts[0] = zs;
        starts[1] = ys;
        starts[2] = xs;

        counts[0] = ze-zs; 
        counts[1] = ye-ys;
        counts[2] = xe-xs;
      
        is_valid = (counts[0] * counts[1] * counts[2] > 0);

        if(is_valid){
          c1_int = oa::utils::make_cube<int>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_int = oa::utils::make_cube<int>(
              A->local_shape());
        
          status = ncmpi_get_vara_int_all(ncid, varid,
                  starts, counts,
                  c2_int.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_int(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_int;        
        
        }else{
          starts[0] = starts[1] = starts[2] = 0;
          counts[0] = counts[1] = counts[2] = 0;
          status = ncmpi_get_vara_int_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      case(NC_FLOAT):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_FLOAT);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);

        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif 
        starts[0] = zs;
        starts[1] = ys;
        starts[2] = xs;

        counts[0] = ze-zs; 
        counts[1] = ye-ys;
        counts[2] = xe-xs;
      
        is_valid = (counts[0] * counts[1] * counts[2] > 0);

        if(is_valid){
          c1_float = oa::utils::make_cube<float>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_float = oa::utils::make_cube<float>(
              A->local_shape());
        
          status = ncmpi_get_vara_float_all(ncid, varid,
                  starts, counts,
                  c2_float.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_float(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_float;        
        
        }else{
          starts[0] = starts[1] = starts[2] = 0;
          counts[0] = counts[1] = counts[2] = 0;
          status = ncmpi_get_vara_float_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      case(NC_DOUBLE):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_DOUBLE);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);

        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif 
        starts[0] = zs;
        starts[1] = ys;
        starts[2] = xs;

        counts[0] = ze-zs; 
        counts[1] = ye-ys;
        counts[2] = xe-xs;
      
        is_valid = (counts[0] * counts[1] * counts[2] > 0);

        if(is_valid){
          c1_double = oa::utils::make_cube<double>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_double = oa::utils::make_cube<double>(
              A->local_shape());
        
          status = ncmpi_get_vara_double_all(ncid, varid,
                  starts, counts,
                  c2_double.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_double(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_double;        
        
        }else{
          starts[0] = starts[1] = starts[2] = 0;
          counts[0] = counts[1] = counts[2] = 0;
          status = ncmpi_get_vara_double_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      default:
        break;
      }

      ncmpi_close(ncid);

      #ifdef __HAVE_CUDA__
      A->memcopy_cpu_to_gpu();
      A->set_newest_buffer(GPU);
      #endif

      return A;

    }

    void save_record(const ArrayPtr& A,
                   const std::string& filename,
                   const std::string& varname,
                   const int record) {

       #ifdef __HAVE_CUDA__
       A->memcopy_gpu_to_cpu();
       A->set_newest_buffer(CPU);
       #endif    
       DataType dt = A->get_data_type();
       int ncid;
       MPI_Comm comm = A->get_partition()->get_comm();
       Shape arr_shape = A->shape();
       int varid;
       int err;
       
       int status;
       
       if (access(filename.c_str(), F_OK) == 0) {
         status = ncmpi_open(comm, filename.c_str(),
                     NC_WRITE, MPI_INFO_NULL, &ncid);
         assert(status == NC_NOERR);
         status = ncmpi_redef(ncid);
         assert(status == NC_NOERR);
       } else {
         status = ncmpi_create(comm, filename.c_str(),
                     NC_64BIT_OFFSET, MPI_INFO_NULL, &ncid);
         assert(status == NC_NOERR);
       }


       if (record == 0){
           int gx = A->shape()[0];
           int gy = A->shape()[1];
           int gz = A->shape()[2];
           int dimid[4];

           //printf("haha\n");

           ncmpi_def_dim(ncid, "x", gx, &dimid[3]);
           ncmpi_def_dim(ncid, "y", gy, &dimid[2]);
           ncmpi_def_dim(ncid, "z", gz, &dimid[1]);
           ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &dimid[0]);
           
           switch(dt) {
           case(DATA_INT):
             err = ncmpi_def_var(ncid, varname.c_str(),
                     NC_INT, 4, dimid, &varid);
             CHECK_ERR(err);
             break;
           case(DATA_FLOAT):
             err = ncmpi_def_var(ncid, varname.c_str(),
                     NC_FLOAT, 4, dimid, &varid);
             CHECK_ERR(err);
             break;
           case(DATA_DOUBLE):
             err = ncmpi_def_var(ncid, varname.c_str(),
                     NC_DOUBLE, 4, dimid, &varid);
             CHECK_ERR(err);
             break;
           }
           ncmpi_enddef(ncid);
        }else{
            
           ncmpi_inq_varid(ncid, varname.c_str(),
                     &varid);
           ncmpi_enddef(ncid);
        }
  
       MPI_Offset start[4], count[4];

       start[3] = A->get_local_box().get_range_x().get_lower();
       start[2] = A->get_local_box().get_range_y().get_lower();
       start[1] = A->get_local_box().get_range_z().get_lower();
       start[0] = record;

       count[3] = A->local_shape()[0];
       count[2] = A->local_shape()[1];
       count[1] = A->local_shape()[2];
       count[0] = 1;
  
       bool has_valid_data = count[1] * count[2] * count[3] > 0;
       
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
       case (DATA_INT):
         if(has_valid_data){
           c_int = utils::make_cube<int>(
               A->buffer_shape(), 
               A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                       arma::span(sw, bsy-sw-1),
                       arma::span(sw, bsz-sw-1));
           
           err = ncmpi_put_vara_int_all(ncid, varid, start,
                   count, c_int.memptr());
           CHECK_ERR(err);
         }else{
           start[0] = start[1] = start[2] = start[3] = 0;
           count[0] = count[1] = count[2] = count[3] = 0;
           err = ncmpi_put_vara_int_all(ncid, varid, start,
                                             count,  NULL);
           CHECK_ERR(err);
         }
         break;
       case (DATA_FLOAT):
         if(has_valid_data){
           c_float = utils::make_cube<float>(
               A->buffer_shape(), 
               A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                       arma::span(sw, bsy-sw-1),
                       arma::span(sw, bsz-sw-1));
           
           err = ncmpi_put_vara_float_all(ncid, varid, start,
                   count, c_float.memptr());
           CHECK_ERR(err);
         }else{
           start[0] = start[1] = start[2] = start[3] = 0;
           count[0] = count[1] = count[2] = count[3] = 0;
           err = ncmpi_put_vara_float_all(ncid, varid, start,
                                             count,  NULL);
           CHECK_ERR(err);
         }
         break;
       case (DATA_DOUBLE):
         if(has_valid_data){
           c_double = utils::make_cube<double>(
               A->buffer_shape(), 
               A->get_buffer(CPU))(arma::span(sw, bsx-sw-1),
                       arma::span(sw, bsy-sw-1),
                       arma::span(sw, bsz-sw-1));
           
           err = ncmpi_put_vara_double_all(ncid, varid, start,
                   count, c_double.memptr());
           CHECK_ERR(err);
         }else{
           start[0] = start[1] = start[2] = start[3] = 0;
           count[0] = count[1] = count[2] = count[3] = 0;
           err = ncmpi_put_vara_double_all(ncid, varid, start,
                                             count,  NULL);
           CHECK_ERR(err);
         }
         break;
       default:
         break;
       }

       ncmpi_close(ncid);
       #ifdef __HAVE_CUDA__
       A->set_newest_buffer(GPU);
       #endif
     }


    ArrayPtr load_record(const std::string& filename, 
                  const std::string& varname,
                  const int record,
                  const MPI_Comm& comm) {
      int status,i;
      int ncid, varid,var_ndims;
      nc_type var_type;
      MPI_Offset *dim_sizes, var_size;
      MPI_Offset *starts, *counts;
      int var_dimids[NC_MAX_VAR_DIMS];
      int ndims,nvars,ngatts,unlimited;
      MPI_Offset gx, gy, gz;
      int sw = Partition::get_default_stencil_width();
      ArrayPtr A;

      int xs, ys, zs, xe, ye, ze;
      cube_int c1_int, c2_int;
      cube_float c1_float, c2_float;
      cube_double c1_double, c2_double;

      void* buf;
      int bsx, bsy, bsz;
      bool is_valid;
      
      status = ncmpi_open(comm, filename.c_str(),
                          NC_NOWRITE, MPI_INFO_NULL, &ncid);

      CHECK_ERR(status);

      status = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &unlimited);

      CHECK_ERR(status);

      dim_sizes = (MPI_Offset*) calloc(ndims, sizeof(MPI_Offset));
   
      for(i=0; i<ndims; i++)  {
          status = ncmpi_inq_dimlen(ncid, i, &(dim_sizes[i]) );
         CHECK_ERR(status);
      }
        
      status = ncmpi_inq_varid(ncid, varname.c_str(), &varid);

      CHECK_ERR(status);

      status = ncmpi_inq_var(ncid, varid, 0, &var_type, &var_ndims,
                             var_dimids, NULL);
      CHECK_ERR(status);

      switch (var_ndims)
          {
            case(VAR_2D):
                if(record < 0){
                  gz = 1;
                  gy = dim_sizes[var_dimids[0]];
                  gx = dim_sizes[var_dimids[1]];
                } else {
                  gz = 1;
                  gy = 1;
                  gx= dim_sizes[var_dimids[1]];
                }
                break;
            case(VAR_3D):
                if(record < 0){
                  gz = dim_sizes[var_dimids[0]];
                  gy = dim_sizes[var_dimids[1]];
                  gx = dim_sizes[var_dimids[2]];
                } else {
                  gz = 1;
                  gy = dim_sizes[var_dimids[1]];
                  gx = dim_sizes[var_dimids[2]];
                }
                break;
             case(VAR_4D):
                gz = dim_sizes[var_dimids[1]];
                gy = dim_sizes[var_dimids[2]];
                gx = dim_sizes[var_dimids[3]];
                break;
             
                
             default:
                gz = 0;
                gy = 0;
                gx = 0;
                break;

          }

      starts = (MPI_Offset*) calloc(var_ndims, sizeof(MPI_Offset));
      counts = (MPI_Offset*) calloc(var_ndims, sizeof(MPI_Offset));
      
      switch(var_type) {
      case(NC_INT):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_INT);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif
        switch (var_ndims)
          {
            case(VAR_2D):
                if(record < 0){
                  starts[0] = ys;
                  starts[1] = xs;
                 
                  counts[0] = ye-ys;
                  counts[1] = xe-xs;
                } else {
                  starts[0] = record;
                  starts[1] = xs;
                 
                  counts[0] = 1;
                  counts[1] = xe-xs;
                }

                is_valid = (counts[0] * counts[1]> 0);
                break;
            case(VAR_3D):
                if(record < 0){
                  starts[0] = zs;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = ze-zs;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
               
                } else {
                  starts[0] = record;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = 1;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
                }
                
                is_valid = (counts[0] * counts[1] * counts[2] > 0);
                break;
             case(VAR_4D):
                starts[0] = record;
                starts[1] = zs;
                starts[2] = ys;
                starts[3] = xs;

                counts[0] = 1;
                counts[1] = ze-zs; 
                counts[2] = ye-ys;
                counts[3] = xe-xs;

                is_valid = (counts[1] * counts[2] * counts[3] > 0);
                break;
                
             default:
                break;
          }

        if(is_valid){
          c1_int = oa::utils::make_cube<int>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_int = oa::utils::make_cube<int>(
              A->local_shape());
        
          status = ncmpi_get_vara_int_all(ncid, varid,
                  starts, counts,
                  c2_int.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_int(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_int;        
        
        }else{
          status = ncmpi_get_vara_int_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      case(NC_FLOAT):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_FLOAT);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif
        switch (var_ndims)
          {
            case(VAR_2D):
                if(record < 0){
                  starts[0] = ys;
                  starts[1] = xs;
                 
                  counts[0] = ye-ys;
                  counts[1] = xe-xs;
                } else {
                  starts[0] = record;
                  starts[1] = xs;
                 
                  counts[0] = 1;
                  counts[1] = xe-xs;
                }

                is_valid = (counts[0] * counts[1]> 0);
                break;
            case(VAR_3D):
                if(record < 0){
                  starts[0] = zs;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = ze-zs;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
               
                } else {
                  starts[0] = record;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = 1;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
                }
                
                is_valid = (counts[0] * counts[1] * counts[2] > 0);
                break;
             case(VAR_4D):
                starts[0] = record;
                starts[1] = zs;
                starts[2] = ys;
                starts[3] = xs;

                counts[0] = 1;
                counts[1] = ze-zs; 
                counts[2] = ye-ys;
                counts[3] = xe-xs;

                is_valid = (counts[1] * counts[2] * counts[3] > 0);
                break;
                
             default:
                break;
          }

        if(is_valid){
          c1_float = oa::utils::make_cube<float>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_float = oa::utils::make_cube<float>(
              A->local_shape());
        
          status = ncmpi_get_vara_float_all(ncid, varid,
                  starts, counts,
                  c2_float.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_float(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_float;        
        
        }else{
          status = ncmpi_get_vara_float_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      case(NC_DOUBLE):
        A = oa::funcs::zeros(comm, {int(gx), int(gy), int(gz)}, sw, DATA_DOUBLE);
        A->get_local_box().get_corners(xs, xe, ys, ye, zs, ze);
        #ifdef __HAVE_CUDA__
        A->memcopy_gpu_to_cpu();
        A->set_newest_buffer(CPU);
        #endif
        switch (var_ndims)
          {
            case(VAR_2D):
                if(record < 0){
                  starts[0] = ys;
                  starts[1] = xs;
                 
                  counts[0] = ye-ys;
                  counts[1] = xe-xs;
                } else {
                  starts[0] = record;
                  starts[1] = xs;
                 
                  counts[0] = 1;
                  counts[1] = xe-xs;
                }

                is_valid = (counts[0] * counts[1]> 0);
                break;
            case(VAR_3D):
                if(record < 0){
                  starts[0] = zs;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = ze-zs;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
               
                } else {
                  starts[0] = record;
                  starts[1] = ys;
                  starts[2] = xs;

                  counts[0] = 1;
                  counts[1] = ye-ys;
                  counts[2] = xe-xs;
                }
                
                is_valid = (counts[0] * counts[1] * counts[2] > 0);
                break;
             case(VAR_4D):
                starts[0] = record;
                starts[1] = zs;
                starts[2] = ys;
                starts[3] = xs;

                counts[0] = 1;
                counts[1] = ze-zs; 
                counts[2] = ye-ys;
                counts[3] = xe-xs;

                is_valid = (counts[1] * counts[2] * counts[3] > 0);
                break;
                
             default:
                break;
          }

        if(is_valid){
          c1_double = oa::utils::make_cube<double>(
              A->buffer_shape(),
              A->get_buffer(CPU));
        
          c2_double = oa::utils::make_cube<double>(
              A->local_shape());
        
          status = ncmpi_get_vara_double_all(ncid, varid,
                  starts, counts,
                  c2_double.memptr());
          CHECK_ERR(status);
      
          bsx = A->buffer_shape()[0];
          bsy = A->buffer_shape()[1];
          bsz = A->buffer_shape()[2];
  
          c1_double(arma::span(sw, bsx-sw-1),
                  arma::span(sw, bsy-sw-1),
                  arma::span(sw, bsz-sw-1)) = c2_double;        
        
        }else{
          status = ncmpi_get_vara_double_all(ncid, varid,
                  starts, counts, NULL);
          CHECK_ERR(status);
        }
        break;
      default:
        break;
      }

      ncmpi_close(ncid);
      #ifdef __HAVE_CUDA__
      A->set_newest_buffer(GPU);
      #endif
      return A;

    }
  }
}
