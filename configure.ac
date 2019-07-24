AC_PREREQ(2.50)
AC_INIT([OpenArray], [1.0])
config_flags="$*"
AM_INIT_AUTOMAKE([subdir-objects no-define no-dependencies foreign])

AS_UNSET([CC])
AS_UNSET([CXX])
AS_UNSET([FC])
AC_PROG_CC([mpicc mpiicc])
AC_PROG_CXX([mpicxx mpiicpc])
AC_PROG_FC([mpif90 mpiifort])

# Append installition path to LDFLAGS and CFLAGS
AC_DEFUN([APPEND_SEARCH_PATH],
        [
            LDFLAGS+=" -L$1/lib "
            CFLAGS+=" -I$1/include "
            CPPFLAGS+=" -I$1/include "
            AC_SUBST(LDFLAGS)
            AC_SUBST(CFLAGS)
            AC_SUBST(CPPFLAGS)
        ])

# Check for MPI LIB
AC_ARG_VAR(MPI_DIR, [Specify the mpi installition root directory which contains the include and lib subdirectory, for example /usr/lib/x86_64-linux-gnu/openmpi])
old_CFLAGS=${CFLAGS}
old_LDFLAGS=${LDFLAGS}
CFLAGS=" -I${MPI_DIR}/include ${CFLAGS}"
LDFLAGS=" -L${MPI_DIR}/lib ${LDFLAGS}"
AC_CHECK_HEADER(mpi.h, [AC_CHECK_FUNC([MPI_Comm_rank], [],
                        dnl maybe -lmpi is needed at link stage
                                     [AC_SEARCH_LIBS([MPI_Comm_rank], [mpi mpich], [],
                                     [AC_MSG_ERROR([
                            -----------------------------------------------------------------------
                                Invalid MPI compiler specified or detected
                                A working MPI C compiler is required. Please specify the location
                                of in the MPI_DIR environment variable. Abort.
                            -----------------------------------------------------------------------])
                        ])])],
                [AC_MSG_ERROR([
                    ----------------------------------------------------------------------------
                    Can not find mpi.h in any existing include path,
                    please consider specify the mpi installition folder using MPI_DIR,
                    then call the configure again, for exmaple:
                    $ ./configure MPI_DIR=/path/to/compilers_and_libraries/linux/mpi/intel64/
                    ----------------------------------------------------------------------------
                    ])])
CFLAGS=${old_CFLAGS}
LDFLAGS=${old_LDFLAGS}
AS_IF([test "x$MPI_DIR" != x], [APPEND_SEARCH_PATH([$MPI_DIR])],
      [])

# check for PnetCDF lib
AC_ARG_VAR(PNETCDF_DIR, [Specify the pnetCdf lib installition root directory which contains the include and lib subdirectory, for example /path/to/pnetcdf_dir/])
old_CFLAGS=${CFLAGS}
old_LDFLAGS=${LDFLAGS}
CFLAGS=" -I${PNETCDF_DIR}/include ${CFLAGS}"
LDFLAGS=" -L${PNETCDF_DIR}/lib ${LDFLAGS}"
echo "CFLAGS = ${CFLAGS}"
AC_CHECK_HEADER(pnetcdf.h, [AC_CHECK_FUNC([ncmpi_open], [],
                                     dnl maybe -lmpi is needed at link stage
                                     [AC_SEARCH_LIBS([ncmpi_open], [pnetcdf], [],
                                     [AC_MSG_ERROR([
                            -----------------------------------------------------------------------
                                Invalid MPI compiler specified or detected
                                A working pdfcdf lib is required. Please specify the location 
                                in the PNETCDF_DIR environment variable. Abort.
                            -----------------------------------------------------------------------])
                        ])])],
                [AC_MSG_ERROR([
                    -----------------------------------------------------------------------------
                    Can not find pnetcdf.h in any existing include path,
                    please consider specify the pnetcdf installition folder using PNETCDF_DIR,
                    then call the configure again, for exmaple:
                    $ ./configure PNETCDF_DIR=/path/to/pnetcdf_dir/
                    -----------------------------------------------------------------------------
                    ])])
CFLAGS=${old_CFLAGS}
LDFLAGS=${old_LDFLAGS}
AS_IF([test "x$PNETCDF_DIR" != x], [APPEND_SEARCH_PATH([$PNETCDF_DIR])],
      [])

AC_SUBST(MPI_DIR)
AC_SUBST(PNETCDF_DIR)

AC_PROG_RANLIB
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
echo "------------------------------------------------------------------------------"
echo \
"
   ${PACKAGE_NAME} Version ${PACKAGE_VERSION}"

echo "\

   Compilers: MPICC    = ${CC}
              MPICXX   = ${CXX}
              FC       = ${FC}
              CPPFLAGS = ${CPPFLAGS}
              CFLAGS   = ${CFLAGS}
              FCFLAGS  = ${FCFLAGS}
              LDFLAGS  = ${LDFLAGS}"

echo "\

   Now run 'make' to build the library and utility tools.
   Then run 'make @<:@<target>@:>@' for testing and installation, where the
   optional <target> can be:
              manual_main - build a test programs (build only, no run)
              install     - install OpenArray library in ${prefix}
              test        - build manual_main and run

------------------------------------------------------------------------------"