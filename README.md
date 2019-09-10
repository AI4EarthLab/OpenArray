# OpenArray V1.0.0
OpenArray is a simple operator library for the decoupling of ocean modelling and parallel computing. The library is promoted as a development tool for the future numerical models to make complex parallel programming transparent. For more details, please refer to our paper (https://www.geosci-model-dev-discuss.net/gmd-2019-28/).

## OpenArray Installation Guide

1.  Getting Started
2.  Alternate Configure Options
3.  Testing the OpenArray installation
4.  Reporting Installation or Usage Problems


### 1. Getting Started

The following instructions take you through a sequence of steps to get the default configuration of OpenArray up and running. **Important note: Please use the same set of compilers to build PnetCDF and OpenArray.** 

(a) You will need the following prerequisites.

```shell
    * The gcc/g++/gfortran compiler, version 4.9.0 or later

    * An MPI C/C++/Fortran compiler, there are three options:
      1) mpich 3.2.1 or later; 2) openmpi v3.0.0 or later; 3) Parallel Studio XE 2017 or later

    * Parallel netCDF version 1.11.2 (http://cucis.ece.northwestern.edu/projects/PnetCDF/)

    * Some basic development tools, including gzip, uzip, make, m4. 
      These are usually part of your operating system's development tools.
```

(b) Specify the MPI compiler.
    For MPICH and Openmpi:

      export MPICC=mpicc  
      export MPICXX=mpicxx  
      export MPIF90=mpif90  
      export MPIF77=mpif77  

   For Intel compiler and Intel MPI library:

      export MPICC=mpiicc  
      export MPICXX=mpiicpc  
      export MPIF90=mpiifort  
      export MPIF77=mpiifort  


(c) Install Parallel netCDF. The default installation directory of PnetCDF is `${HOME}/install`:
     
      cd
      wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/pnetcdf-1.11.2.tar.gz
      tar xf pnetcdf-1.11.2.tar.gz
      cd pnetcdf-1.11.2
      ./configure --prefix=${HOME}/install  
      make 
      make install 


(d) Install OpenArray. The default installation directory of OpenArray is `${HOME}/install`:

      wget https://github.com/hxmhuang/OpenArray/archive/master.zip
      unzip master.zip
      cd OpenArray-master
      ./configure --prefix=${HOME}/install  PNETCDF_DIR=${HOME}/install   
      make (make -j8 for parallel make)
      make install
      ./manual_main

   This executable program `manual_main` is a demo written based on OpenArray.
   If you have completed all of the above steps, you have successfully installed OpenArray.
      

### 2. Alternate Configure Options

OpenArray has a number of configure features.  A complete list of configuration
options can be found using:

   ./configure --help

    Here lists a few important options:

     --prefix=PREFIX      install OpenArray files in PREFIX [/usr/local]
     --with-mpi=/path/to/implementation
                          The installation prefix path for MPI implementation.
    Some influential environment variables:
     MPICC       MPI C compiler, [default: CC]
     MPICXX      MPI C++ compiler, [default: CXX]
     MPIF77      MPI Fortran 77 compiler, [default: F77]
     MPIF90      MPI Fortran 90 compiler, [default: FC]
     CC          C compiler command
     CFLAGS      C compiler flags
     LDFLAGS     linker flags, e.g. -L<lib dir> if you have libraries in a
                 nonstandard directory <lib dir>
     LIBS        libraries to pass to the linker, e.g. -l<library>
     CPPFLAGS    (Objective) C/C++ preprocessor flags, e.g. -I<include dir> if
                  you have headers in a nonstandard directory <include dir>
     CXX         C++ compiler command
     CXXFLAGS    C++ compiler flags
     FC          Fortran compiler command
     FCFLAGS     Fortran compiler flags
     PNETCDF_DIR Specify the pnetCdf lib installition root directory which
                 contains the include and lib subdirectory, for example
                 /path/to/pnetcdf_dir/
     CPP         C preprocessor


### 3. Testing the OpenArray installation

For testing OpenArray, the command is:
      
     make test

### 4. Reporting Installation or Usage Problems

Please report the problems on our github: https://github.com/hxmhuang/OpenArray/issues



