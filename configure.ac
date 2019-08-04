AC_PREREQ(2.50)
AC_INIT([OpenArray], [1.0])
config_flags="$*"
AM_INIT_AUTOMAKE([subdir-objects no-define no-dependencies foreign])

AC_PROG_GREP
AC_PROG_EGREP
AC_CHECK_PROGS(RM, [rm])
AC_CHECK_PROGS(TIME, [time])
AC_CHECK_PROGS(NPROC_PROG, [nproc])
# Append installition path to LDFLAGS and CFLAGS
AC_DEFUN([APPEND_SEARCH_PATH],
        [
            LDFLAGS+=" -L$1/lib "
            CFLAGS+=" -I$1/include "
            FCFLAGS+=" -I$1/include "
            CPPFLAGS+=" -I$1/include "
            CXXFLAGS+=" -I$1/include "
            AC_SUBST(LDFLAGS)
            AC_SUBST(CFLAGS)
            AC_SUBST(FCFLAGS)
            AC_SUBST(CPPFLAGS)
            AC_SUBST(CXXFLAGS)
        ])

AC_DEFUN([UD_MSG_DEBUG],
         [if test "x${ac_debug}" = xyes ; then
             AC_MSG_NOTICE([DEBUG: $1])
          fi
         ]
)

AC_ARG_ENABLE(cxx,
    [AS_HELP_STRING([--disable-cxx],
                    [Turn off support for the C++ interface,
                     if you only need the C interface. @<:@default: enabled@:>@])],
    [enable_cxx=${enableval}], [enable_cxx=auto]
)
UD_MSG_DEBUG(enable_cxx=$enable_cxx)

MPI_INSTALL=
AC_ARG_WITH(mpi,
   [AS_HELP_STRING([--with-mpi=/path/to/implementation],
                   [The installation prefix path for MPI implementation.])],
   [if test "x${withval}" = x ; then
       AC_MSG_ERROR(--with-mpi is set but the value is NULL)
    fi
    if (! test -d "${withval}") ; then
      AC_MSG_ERROR(Directory '${withval}' specified in --with-mpi does not exist)
    fi
    APPEND_SEARCH_PATH([$withval])
    MPI_INSTALL=${withval}]
)
dnl MPI_INSTALL will be referred in UD_MPI_PATH_PROGS and UD_MPI_PATH_PROG
dnl defined in acinclude.m4

AC_ARG_VAR(MPICC,  [MPI C compiler, @<:@default: CC@:>@])
AC_ARG_VAR(MPICXX, [MPI C++ compiler, @<:@default: CXX@:>@])
AC_ARG_VAR(MPIF77, [MPI Fortran 77 compiler, @<:@default: F77@:>@])
AC_ARG_VAR(MPIF90, [MPI Fortran 90 compiler, @<:@default: FC@:>@])

dnl Check if MPICC, MPICXX, MPIF77, MPIF90 are set by the user.
dnl If not, set MPICC equal to CC. Similarly, for MPICXX, MPIF77, and MPIF90
ac_user_MPICC=$MPICC
ac_user_MPICXX=$MPICXX
ac_user_MPIF77=$MPIF77
ac_user_MPIF90=$MPIF90
if test "x$MPICC"  = x -a "x$CC"  != x ; then ac_user_MPICC=$CC   ; fi
if test "x$MPICXX" = x -a "x$CXX" != x ; then ac_user_MPICXX=$CXX ; fi
if test "x$MPIF77" = x -a "x$F77" != x ; then ac_user_MPIF77=$F77 ; fi
if test "x$MPIF77" = x -a "x$FC"  != x ; then ac_user_MPIF77=$FC  ; fi
if test "x$MPIF90" = x -a "x$F90" != x ; then ac_user_MPIF90=$F90 ; fi
if test "x$MPIF90" = x -a "x$FC"  != x ; then ac_user_MPIF90=$FC  ; fi

dnl add Intel MPI compiler wrappers
CANDIDATE_MPICC="  ${MPICC} mpiicc icc"
CANDIDATE_MPICXX=" ${MPICXX} mpiicpc mpiicxx mpiic++ mpiiCC icpc"
CANDIDATE_MPIF77=" ${MPIF77} mpiifort mpiifc mpiif77 ifort"
CANDIDATE_MPIF90=" ${MPIF90} mpiifort mpiifc mpiif95 mpiif90 ifort"

CANDIDATE_MPICC+="   mpicc mpicc_r"
CANDIDATE_MPICXX+=" mpicxx mpic++ mpiCC mpcxx mpc++ mpicxx_r mpiCC_r mpcxx_r mpic++_r mpc++_r"
CANDIDATE_MPIF77+=" mpif77 mpif77_r mpf77 mpf77_r"
CANDIDATE_MPIF90+=" mpif90 mpif90_r mpf90 mpf90_r mpif95 mpif95_r mpf95 mpf95_r"

dnl add GNU MPI compilers
CANDIDATE_MPICC+=" mpigcc mpgcc mpigcc_r mpgcc_r"
CANDIDATE_MPICXX+=" mpig++ mpg++ mpig++_r mpg++_r"
CANDIDATE_MPIF77+=" mpig77 mpig77_r mpg77 mpg77_r"
CANDIDATE_MPIF90+=" mpigfortran mpgfortran mpigfortran_r mpgfortran_r"

dnl add IBM MPI compilers
CANDIDATE_MPICC+=" mpcc_r mpcc mpixlc_r mpixlc"
CANDIDATE_MPICXX+=" mpCC_r mpCC mpixlcxx_r mpixlcxx mpixlC_r mpixlC"
CANDIDATE_MPIF77+=" mpixlf77_r mpixlf77"
CANDIDATE_MPIF90+=" mpixlf90_r mpixlf90"

dnl add IBM BGL MPI compilers
CANDIDATE_MPICC+=" blrts_xlc mpxlc_r mpxlc"
CANDIDATE_MPICXX+=" blrts_xlC mpxlC_r mpxlC mpixlc++ mpxlcxx mpxlc++ mpxlCC mpixlc++_r mpxlcxx_r mpxlc++_r mpxlCC_r"
CANDIDATE_MPIF77+=" blrts_xlf mpxlf_r mpxlf mpxlf77 mpxlf77_r mpixlf mpixlf_r"
CANDIDATE_MPIF90+=" blrts_xlf90 mpxlf90_r mpxlf90 mpxlf95_r mpxlf95 mpixlf95 mpixlf95_r"

dnl add Fujitsu MPI compilers
CANDIDATE_MPICC+=" mpifccpx"
CANDIDATE_MPICXX+=" mpiFCCpx"
CANDIDATE_MPIF77+=" mpifrtpx"
CANDIDATE_MPIF90+=" mpifrtpx"

dnl add Cray MPI compiler wrappers
CANDIDATE_MPICC+=" cc"
CANDIDATE_MPICXX+=" CC"
CANDIDATE_MPIF77+=" ftn"
CANDIDATE_MPIF90+=" ftn"

dnl add PGI MPI compiler wrappers
CANDIDATE_MPICC+=" mpipgcc mppgcc"
CANDIDATE_MPICXX+=" mpipgCC mppgCC"
CANDIDATE_MPIF77+=" mpipgf77 mppgf77"
CANDIDATE_MPIF90+=" mpipgf90 mpipgf95 mppgf90 mppgf95 "

dnl find the full path of MPICC from CANDIDATE_MPICC and MPI_INSTALL
if test "x${ac_user_MPICC}" = x ; then
   dnl if MPICC or CC has not been set by users, then search from
   dnl CANDIDATE_MPICC, and find the full path of MPICC
   UD_MPI_PATH_PROGS([MPICC], [$CANDIDATE_MPICC])
else
   dnl check whether user specified MPICC is valid
   UD_MPI_PATH_PROG([MPICC], [$ac_user_MPICC])
fi

if test "x${MPICC}" = x ; then
   if test "x$ac_user_MPICC" = x ; then
      ERR_MSG="No MPI C compiler can be found"
   else
      ERR_MSG="Specified MPI C compiler \"$ac_user_MPICC\" cannot be found"
   fi
   if test "x$MPI_INSTALL" != x ; then
      ERR_MSG+=" under $MPI_INSTALL"
   fi
   AC_MSG_ERROR([
   -----------------------------------------------------------------------
     $ERR_MSG
     OpenArray requires a working MPI C compiler. Please specify the location
     of an MPI C compiler, either in the MPICC environment variable
     (not CC variable) or through --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
fi
UD_MSG_DEBUG([MPICC=$MPICC])

dnl Now MPICC is configured, do some basic compiler tests
CC=${MPICC}
AC_PROG_CC
dnl AM_PROG_CC_C_O

dnl Set output variable CXX to a command that runs the C preprocessor.
dnl Some C compilers require -E to be used as C preprocessor.
CXX=${MPICXX}
AC_PROG_CXX

dnl check if MPICC works for basic MPI call: MPI_Comm_rank()
AC_CHECK_FUNC([MPI_Comm_rank], [],
   dnl maybe -lmpi is needed at link stage
   [AC_SEARCH_LIBS([MPI_Comm_rank], [mpi mpich], [],
                 [AC_MSG_ERROR([
   -----------------------------------------------------------------------
     Invalid MPI compiler specified or detected: "${MPICC}"
     A working MPI C compiler is required. Please specify the location
     of one either in the MPICC environment variable (not CC variable) or
     through --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
])])

AC_CHECK_FUNC([MPI_File_open], [],
   dnl maybe -lmpio is needed at link stage
   [AC_SEARCH_LIBS([MPI_File_open], [mpio], [],
                   [AC_MSG_ERROR([
   -----------------------------------------------------------------------
     The underneath MPI implementation does not support MPI-IO.
     PnetCDF requires MPI-IO support to work properly. Abort.
   -----------------------------------------------------------------------])
])])

dnl check the base of MPICC compiler
UD_CHECK_MPICC_BASE
AM_CONDITIONAL(MPICC_IS_XLC,   [test "x${ac_cv_mpicc_base}" = xXLC])
AM_CONDITIONAL(MPICC_IS_PGCC,  [test "x${ac_cv_mpicc_base}" = xPGCC])
AM_CONDITIONAL(MPICC_IS_FCCPX, [test "x${ac_cv_mpicc_base}" = xFCCPX])

dnl print out MPI version and vendor infomration
CHECK_MPI_VERSION

dnl compute canonical system types
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

dnl AM_PROG_AR is first recognized/traced in autoconf 2.68
dnl Use it only if we want support for unusual archivers such as Microsoft
dnl lib.. AM_PROG_AR must be called before LT_INIT
AM_PROG_AR

dnl check MPI C++ compiler
AC_ARG_ENABLE(cxx,
    [AS_HELP_STRING([--disable-cxx],
                    [Turn off support for the C++ interface,
                     if you only need the C interface. @<:@default: enabled@:>@])],
    [enable_cxx=${enableval}], [enable_cxx=auto]
)
UD_MSG_DEBUG(enable_cxx=$enable_cxx)

dnl If MPICXX or CXX has not been set by users, then
dnl search from CANDIDATE_MPICXX and find the full path of MPICXX
if test "x${enable_cxx}" != xno ; then
   if test "x${ac_user_MPICXX}" = x ; then
      dnl if MPICXX or CXX has not been set by users, then search from
      dnl CANDIDATE_MPICXX, and find the full path of MPICXX
      UD_MPI_PATH_PROGS([MPICXX], [$CANDIDATE_MPICXX])
   else
      dnl check whether user specified MPICXX is valid
      UD_MPI_PATH_PROG([MPICXX], [$ac_user_MPICXX])
   fi
   UD_MSG_DEBUG([MPICXX=$MPICXX])

   if test "x${MPICXX}" = x ; then
      if test "x$ac_user_MPICXX" = x ; then
         ERR_MSG="No MPI C++ compiler can be found"
      else
         ERR_MSG="Specified MPI C++ compiler \"$ac_user_MPICXX\" cannot be found"
      fi
      if test "x$MPI_INSTALL" != x ; then
         ERR_MSG+=" under $MPI_INSTALL"
      fi
      if test "x${enable_cxx}" = xyes ; then
         dnl --enable-cxx is explicitly set at command line
         AC_MSG_ERROR([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Please specify the location of the MPI C++ compiler, either in the
     MPICXX environment variable or the --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
      else
         dnl enable_cxx is auto, i.e. --enable-cxx is not set at command line
         AC_MSG_WARN([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Thus the C++ feature is disabled.
     The location of the MPI C++ compiler can be specified either in the
     MPICXX environment variable or through the --with-mpi configure flag.
   -----------------------------------------------------------------------])
      fi
   fi
fi

if test "x${enable_cxx}" = xno -o "x${MPICXX}" = x ; then
   dnl when cxx is explicitly disabled or no MPICXX can be found
   has_mpicxx=no
else
   has_mpicxx=yes
   CXX=${MPICXX}
fi

dnl Must invoke AC_PROG_CXX here, because AC_PROG_CXX cannot be called
dnl conditionally in Automake, otherwise got "error: conditional
dnl "am__fastdepCXX" was never defined." So, if has_mpicxx is no, then
dnl AC_PROG_CXX will search for g++. However, g++ will not be used at all.
AC_PROG_CXX

dnl autoconf 2.59 has not yet implemented AC_PROG_CXX_C_O
dnl AC_PROG_CXX_C_O

if test "x${has_mpicxx}" = xyes ; then
   dnl test if MPICXX can compile an MPI-IO program
   AC_LANG_PUSH(C++)
   AC_CHECK_FUNC([MPI_File_close], [],
      dnl maybe -lmpi++ is needed at link stage
      [AC_SEARCH_LIBS([MPI_File_close], [mpi++ mpichcxx mpi_cxx], [],
                      [has_mpicxx=no])])
   AC_LANG_POP(C++)
   if test "x$has_mpicxx" = xno ; then
      if test "x${enable_cxx}" = xyes ; then
         AC_MSG_ERROR([
   -----------------------------------------------------------------------
     Explicitly requesting C++ feature, but "${MPICXX}"
     is not a working MPI C++ compiler. Abort.
   -----------------------------------------------------------------------])
      else dnl in case enable_cxx is auto
         AC_MSG_WARN([
   -----------------------------------------------------------------------
     "${MPICXX}" is not a working MPI C++ compiler.
     Thus, the C++ feature is disabled.
   -----------------------------------------------------------------------])
      fi
   else
      UD_CHECK_MPI_CPP_SEEK_SET
      UD_MSG_DEBUG(ac_cv_CHECK_MPI_CPP_SEEK_SET=$ac_cv_CHECK_MPI_CPP_SEEK_SET)
      dnl When using older version Intel compilers 4.x, SEEK_SET will be
      dnl reported as redefined. We need to add the following C++ preprocessor
      dnl flags. See doc/README.INTEL
      if test "x${ac_cv_CHECK_MPI_CPP_SEEK_SET}" = xyes ; then
         SEEK_SET_REDEFINED=yes
      fi
   fi
fi
AC_SUBST(has_mpicxx)dnl for src/utils/pnetcdf-config.in
UD_MSG_DEBUG(has_mpicxx=$has_mpicxx)
AM_CONDITIONAL(HAS_MPICXX, [test x$has_mpicxx = xyes])
AM_CONDITIONAL(SEEK_SET_REDEFINED, [test x$ac_cv_CHECK_MPI_CPP_SEEK_SET = xyes])


AC_ARG_ENABLE(fortran,
    [AS_HELP_STRING([--disable-fortran],
                    [Turn off support for the Fortran interface,
                     if you only need the C interface. @<:@default: enabled@:>@])],
    [enable_fortran=${enableval}], [enable_fortran=auto]
)

if test "x${enable_fortran}" = xno ; then
   has_fortran=no
else
   has_fortran=yes
   UD_MPI_PATH_PROGS([MPIF77], [$CANDIDATE_MPIF77])

   if test "x${MPIF77}" = x ; then
      if test "x$ac_user_MPIF77" = x ; then
         ERR_MSG="No MPI Fortran 77 compiler can be found"
      else
         ERR_MSG="Specified MPI Fortran 77 compiler \"$ac_user_MPIF77\" cannot be found"
      fi
      if test "x$MPI_INSTALL" != x ; then
         ERR_MSG+=" under $MPI_INSTALL"
      fi
      if test "x${enable_fortran}" = xyes ; then
         dnl --enable-fortran is explicitly set at command line
         AC_MSG_ERROR([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Please specify the location of the MPI Fortran 77 compiler, either in the
     MPIF77 environment variable or the --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
      else
         dnl enable_fortran is auto, i.e. --enable-fortran is not set at command line
         AC_MSG_WARN([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Thus, the Fortran feature is disabled.
     The location of the MPI Fortran 77 compiler can be specified either in the
     MPIF77 environment variable or through the --with-mpi configure flag.
   -----------------------------------------------------------------------])
      fi
   fi
fi

if test "x${has_fortran}" = xyes ; then
      dnl if MPIF90 or F90 has not been set by users, then search from
      dnl CANDIDATE_MPIF90, and find the full path of MPIF90
      UD_MPI_PATH_PROGS([MPIF90], [$CANDIDATE_MPIF90])

   if test "x${MPIF90}" = x ; then
      has_fortran=no
      if test "x$ac_user_MPIF90" = x ; then
         ERR_MSG="No MPI Fortran 90 compiler can be found"
      else
         ERR_MSG="Specified MPI Fortran 90 compiler \"$ac_user_MPIF90\" cannot be found"
      fi
      if test "x$MPI_INSTALL" != x ; then
         ERR_MSG+=" under $MPI_INSTALL"
      fi
      if test "x${enable_fortran}" = xyes ; then
         dnl --enable-fortran is explicitly set at command line
         AC_MSG_ERROR([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Please specify the location of the MPI Fortran 90 compiler, either in the
     MPIF90 environment variable or the --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
      else
         dnl enable_fortran is auto, i.e. --enable-fortran is not set at command line
         AC_MSG_WARN([
   -----------------------------------------------------------------------
     $ERR_MSG.
     Thus, the Fortran feature is disabled.
     The location of the MPI Fortran 90 compiler can be specified either in the
     MPIF90 environment variable or through the --with-mpi configure flag.
   -----------------------------------------------------------------------])
      fi
   fi
fi
UD_MSG_DEBUG(has_fortran=${has_fortran})
UD_MSG_DEBUG([MPIF77=$MPIF77])
UD_MSG_DEBUG([MPIF90=$MPIF90])

mpi_mod=no
if test "x${has_fortran}" = xyes ; then
   dnl Check if MPIF90 is a valid MPI compiler
   FC=${MPIF90}
   AC_PROG_FC
   dnl FCFLAGS is set in AC_PROG_FC

   dnl FCLIBS_save="$FCLIBS"
   dnl FCLIBS=""
   AC_FC_LIBRARY_LDFLAGS
   dnl UD_MSG_DEBUG([before FCLIBS=$FCLIBS])
   dnl The autoconf macro for finding FCLIBS sometimes makes mistakes
   dnl (particularly with the Fujitsu frt compiler).  This next step
   dnl first sees if the FCLIBS is valid with the Fortran compiler
   dnl This also happens to Solaris Studio Fortran compilers
   dnl AC_PROG_FC_FCLIBS_VALID
   dnl Now see if FCLIBS works with the C compiler (remove invalid ones)
   dnl PAC_PROG_FC_CHECK_FCLIBS

   dnl replace FLIBS and F90LIBS with FCLIBS
   dnl UD_MSG_DEBUG([after FCLIBS=$FCLIBS])
   dnl FLIBS="$FCLIBS"
   dnl F90LIBS="$FCLIBS"
   AC_MSG_RESULT($FCFLAGS)

   AC_MSG_CHECKING([whether mpi.mod is available])
   AC_LANG_PUSH([Fortran])
   #  AC_COMPILE_IFELSE([AC_LANG_PROGRAM([], [[        use mpi]])],
   #                    [mpi_mod=yes], [mpi_mod=no]
   #  )
   mpi_mod=yes
   AC_MSG_RESULT($mpi_mod)
   AC_MSG_CHECKING([whether $MPIF90 is a valid MPI compiler])
   AC_LANG_POP([Fortran])
   AC_MSG_RESULT($valid_mpif90)

   if test "x${valid_mpif90}" = xno ; then
      has_fortran=no
      if test "x${enable_fortran}" = xyes ; then
         AC_MSG_ERROR([
   -----------------------------------------------------------------------
     Invalid MPI Fortran 90 compiler: "${MPIF90}"
     A working MPI Fortran 90 compiler is required. Please specify the
     location of a valid MPI Fortran 90 compiler, either in the MPIF90
     environment variable or through --with-mpi configure flag. Abort.
   -----------------------------------------------------------------------])
      fi
      AC_MSG_WARN([
   -----------------------------------------------------------------------
     "${MPIF90}" is not a working MPI Fortran 90 compiler.
     Thus, the Fortran feature is disabled.
   -----------------------------------------------------------------------])
   else
      if test "x${mpi_mod}" = xyes ; then
         USE_MPIF_HEADER="use mpi, only: MPI_OFFSET_KIND"
      else
         USE_MPIF_HEADER="include \"mpif.h\""
      fi
      AC_SUBST(USE_MPIF_HEADER)
   fi
fi
AC_SUBST(has_fortran)dnl for src/utils/pnetcdf-config.in
AM_CONDITIONAL(HAS_FORTRAN, [test x$has_fortran = xyes])
AM_CONDITIONAL(HAVE_MPI_MOD, [test x$mpi_mod = xyes])

if test "x${has_fortran}" = xyes ; then
   if test "x${enable_strict}" = xyes -a "x${GFC}" = xyes; then
      dnl not all Fortran compilers recognize -Wall
      FFLAGS+=" -Wall"
      FCFLAGS+=" -Wall"
   fi
   dnl AC_SUBST(FLIBS)
   dnl AC_SUBST(FCLIBS)
   dnl AC_SUBST(F90LIBS)
   dnl AC_SUBST(FLDFLAGS)
   dnl AC_SUBST(F90LDFLAGS)
fi

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
                                A working pnetcdf lib is required. Please specify the location 
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

AC_CANONICAL_HOST
case "${host_os}" in
    darwin*)
         CFLAGS+=" -D_OSX_ "
         AC_SUBST(CFLAGS)
         FCFLAGS+=" -D_OSX_ "
         AC_SUBST(FCFLAGS)
         CPPFLAGS+=" -D_OSX_ "
         AC_SUBST(CPPFLAGS)
         CXXFLAGS+=" -D_OSX_ "
         AC_SUBST(CXXFLAGS)
        ;;
    *)
        ;;
esac
AM_CONDITIONAL([HAS_NPROC], [test "x$NPROC_PROG" != x])

AC_SUBST(MPI_DIR)
AC_SUBST(PNETCDF_DIR)

AC_PROG_RANLIB
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
echo "------------------------------------------------------------------------------"
echo \
"
   ${PACKAGE_NAME} Version ${PACKAGE_VERSION}

   Features:  Build Fortran APIs                          - ${has_fortran}"

if test "x${has_fortran}" = xyes -a "x${ac_f77_support_freeform}" = xno ; then
   echo "\
              Support free form in Fortran 77             - disabled"
fi

echo "\

   Compilers: MPICC    = ${MPICC}"
if test "${has_mpicxx}" = yes ; then
   echo "\
              MPICXX   = ${MPICXX}"
fi
if test "${has_fortran}" = yes ; then
   echo "\
              MPIF77   = ${MPIF77}
              MPIF90   = ${MPIF90}"
fi
if test "x${CPPFLAGS}" != x ; then
   echo "\
              CPPFLAGS = ${CPPFLAGS}"
fi
echo "\
              CFLAGS   = ${CFLAGS}"
if test "${has_mpicxx}" = yes ; then
   echo "\
              CXXFLAGS = ${CXXFLAGS}"
fi
if test "${has_fortran}" = yes ; then
   echo "\
              FCFLAGS  = ${FCFLAGS}"
fi
if test "x${LDFLAGS}" != x ; then
   echo "\
              LDFLAGS  = ${LDFLAGS}"
fi
if test "x${LIBS}" != x ; then
   echo "\
              LIBS     = ${LIBS}"
fi

echo "\

   Now run 'make' to build the library and utility tools.
   Then run 'make @<:@<target>@:>@' for testing and installation, where the
   optional <target> can be:
              manual_main - build a test programs (build only, no run)
              install     - install OpenArray library in ${prefix}
              test        - build manual_main and run

------------------------------------------------------------------------------"
