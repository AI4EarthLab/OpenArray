# OpenArray V1.0.0
OpenArray is a simple operator library for the decoupling of ocean modelling and parallel computing. The library is promoted as a development tool for the future numerical models to make complex parallel programming transparent. For more details, please refer to our paper (https://www.geosci-model-dev-discuss.net/gmd-2019-28/).

# OpenArray Installation Guide

1.  Getting Started
2.  Alternate Configure Options
3.  Testing the OpenArray installation
4.  Reporting Installation or Usage Problems


## 1. Getting Started

The following instructions take you through a sequence of steps to get the default configuration of OpenArray up and running.

(a) You will need the following prerequisites.

    - REQUIRED: This installation package

    - REQUIRED: The gcc/g++/gfortran compiler, version 4.9.0 or latter

    - REQUIRED: An MPI C/C++/Fortran compiler

    - REQUIRED: Parallel netCDF (http://cucis.ece.northwestern.edu/projects/PnetCDF/)

    - REQUIRED: Some other development tools, including gzip, make, m4, automake.
                These are usually part of your operating system's development tools.

(b) Unzip the package and go to the top level directory:

      wget https://github.com/hxmhuang/OpenArray/archive/master.zip -O OpenArray-master.zip
      unzip OpenArray-master.zip
      cd OpenArray-master

(c) Configure OpenArray specifying the installation directory, e.g. ${HOME}/install 

      ./configure --prefix=${HOME}/install

   If the Parallel netCDF or MPI compilers are not configured in the default
   environment variables, you may specify these configurations, for example:

      MPICC=mpicc MPICXX=mpicxx MPIF90=mpif90 PNETCDF_DIR=${where you install pnetcdf} ./configure --prefix=${HOME}/install

(d) Build OpenArray:

      make

   Or try parallel make, e.g. (using 8 simultaneous jobs)

      make -j8

(e) Install OpenArray and test

      make install
      ./manual_main

   This executable program `manual_main` is a demo written based on OpenArray.
   If you have completed all of the above steps, you have successfully installed OpenArray.


## 2. Alternate Configure Options

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


## 3. Testing the OpenArray installation

For testing OpenArray, the command is
     make test


## 4. Reporting Installation or Usage Problems

Please report the problems on our github: https://github.com/hxmhuang/OpenArray/issues



