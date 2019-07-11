      PROGRAM PDPBLASDRIVER
*
*  -- PBLAS example code --
*     University of Tennessee, Knoxville, Oak Ridge National Laboratory,
*     and University of California, Berkeley.
*
*     Written by Antoine Petitet, August 1995 (petitet@cs.utk.edu)
*
*     This program shows how to set the matrix descriptors and call
*     the PBLAS routines.
*
*     .. Parameters ..
      INTEGER            DBLESZ, MEMSIZ, TOTMEM
      PARAMETER          ( DBLESZ = 8, TOTMEM = 2000000,
     $                     MEMSIZ = TOTMEM / DBLESZ )
      INTEGER            BLOCK_CYCLIC_2D, CSRC_, CTXT_, DLEN_, DT_,
     $                   LLD_, MB_, M_, NB_, N_, RSRC_
      PARAMETER          ( BLOCK_CYCLIC_2D = 1, DLEN_ = 9, DT_ = 1,
     $                     CTXT_ = 2, M_ = 3, N_ = 4, MB_ = 5, NB_ = 6,
     $                     RSRC_ = 7, CSRC_ = 8, LLD_ = 9 )
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      CHARACTER*80       OUTFILE
      INTEGER            IAM, IASEED, IBSEED, ICSEED, ICTXT, INFO, IPA,
     $                   IPB, IPC, IPW, K, KP, KQ, M, MP, MYCOL, MYROW,
     $                   N, NB, NOUT, NPCOL, NPROCS, NPROW, NQ, WORKSIZ
      DOUBLE PRECISION   BNRM2
*     ..
*     .. Local Arrays ..
      INTEGER            DESCA( DLEN_ ), DESCB( DLEN_ ), DESCC( DLEN_ )
      DOUBLE PRECISION   MEM( MEMSIZ )
*     ..
*     .. External Subroutines ..
      EXTERNAL           BLACS_EXIT, BLACS_GET, BLACS_GRIDEXIT,
     $                   BLACS_GRIDINFO, BLACS_GRIDINIT, BLACS_PINFO,
     $                   DESCINIT, IGSUM2D, PDMATGEN, PDPBLASINFO,
     $                   PDNRM2, PDGEMV, PDGEMM, PDLAPRNT
*     ..
*     .. External Functions ..
      INTEGER            NUMROC
      EXTERNAL           NUMROC
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DBLE, MAX
*     ..
*     .. Executable Statements ..
*
*     Get starting information
*
      CALL BLACS_PINFO( IAM, NPROCS )

      print*, "OUTFILE", trim(OUTFILE)

c$$$      print*, "NOUT=", NOUT, "M=", M, "N=", N, "K=", K, 
c$$$     $     "NB=", NB, "NPROW=", NPROW, "NPCOL=", NPCOL, 
c$$$     $     "IAM=", IAM, "NPROCS=", NPROCS, "shape(MEM)=", shape(MEM)
c$$$      
c$$$      CALL PDPBLASINFO( OUTFILE, NOUT, M, N, K, NB, NPROW, NPCOL, MEM,
c$$$     $      IAM, NPROCS )
c$$$
c$$$      print*, "NOUT=", NOUT, "M=", M, "N=", N, "K=", K, 
c$$$     $     "NB=", NB, "NPROW=", NPROW, "NPCOL=", NPCOL, 
c$$$     $     "IAM=", IAM, "NPROCS=", NPROCS,"shape(MEM)=", shape(MEM)

      M = 4;
      N=4;
      K=4;
      NB=2;
      NPROW=2;
      NPCOL=2;
!      allocate(MEM(250000))
*
*     Define process grid
*
      CALL BLACS_GET( -1, 0, ICTXT )
      print*, "ICTXT=", ICTXT
      CALL BLACS_GRIDINIT( ICTXT, 'Row-major', NPROW, NPCOL )
      print*, "ICTXT=", ICTXT      
      CALL BLACS_GRIDINFO( ICTXT, NPROW, NPCOL, MYROW, MYCOL )
      print*, "myrow=", myrow, "mycol=", mycol
      print*, "size(dESCA)=", shape(DESCA)
*
*     Go to bottom of process grid loop if this case doesn't use my
*     process
*
c$$$      IF( MYROW.GE.NPROW .OR. MYCOL.GE.NPCOL )
c$$$     $   GO TO 20
*
      MP = NUMROC( M, NB, MYROW, 0, NPROW )
      KP = NUMROC( K, NB, MYROW, 0, NPROW )
      KQ = NUMROC( K, NB, MYCOL, 0, NPCOL )
      NQ = NUMROC( N, NB, MYCOL, 0, NPCOL )
*
*     Initialize the array descriptor for the matrix A, B and C
*
      CALL DESCINIT( DESCA, M, K, NB, NB, 0, 0, ICTXT, MAX( 1, MP ),
     $               INFO )
      CALL DESCINIT( DESCB, K, N, NB, NB, 0, 0, ICTXT, MAX( 1, KP ),
     $               INFO )
      CALL DESCINIT( DESCC, M, N, NB, NB, 0, 0, ICTXT, MAX( 1, MP ),
     $     INFO )
      PRINT*, "DESCA = ", DESCA

      CALL PDLAPRNT( M, K, MEM, 1, 1, DESCA, 0, 0,
     $     'A', NOUT, MEM )
      STOP
*
*     Assign pointers into MEM for SCALAPACK arrays, A is
*     allocated starting at position MEM( 1 )
*
      IPA = 1
      IPB = IPA + DESCA( LLD_ )*KQ
      IPC = IPB + DESCB( LLD_ )*NQ
      IPW = IPC + DESCC( LLD_ )*NQ
*
      WORKSIZ = NB
*
*     Check for adequate memory for problem size
*
c$$$      INFO = 0
c$$$      IF( IPW+WORKSIZ.GT.MEMSIZ ) THEN
c$$$         IF( IAM.EQ.0 )
c$$$     $      WRITE( NOUT, FMT = 9998 ) 'test', ( IPW+WORKSIZ )*DBLESZ
c$$$         INFO = 1
c$$$      END IF
*
*     Check all processes for an error
*
c$$$      CALL IGSUM2D( ICTXT, 'All', ' ', 1, 1, INFO, 1, -1, 0 )
c$$$      IF( INFO.GT.0 ) THEN
c$$$         IF( IAM.EQ.0 )
c$$$     $      WRITE( NOUT, FMT = 9999 ) 'MEMORY'
c$$$         GO TO 10
c$$$      END IF
*
*     Generate random matrices A, B and C
*
c$$$      IASEED = 100
c$$$      CALL PDMATGEN( ICTXT, 'No transpose', 'No transpose', DESCA( M_ ),
c$$$     $               DESCA( N_ ), DESCA( MB_ ), DESCA( NB_ ),
c$$$     $               MEM( IPA ), DESCA( LLD_ ), DESCA( RSRC_ ),
c$$$     $               DESCA( CSRC_ ), IASEED, 0, MP, 0, KQ, MYROW, MYCOL,
c$$$     $               NPROW, NPCOL )
c$$$      IBSEED = 200
c$$$      CALL PDMATGEN( ICTXT, 'No transpose', 'No transpose', DESCB( M_ ),
c$$$     $               DESCB( N_ ), DESCB( MB_ ), DESCB( NB_ ),
c$$$     $               MEM( IPB ), DESCB( LLD_ ), DESCB( RSRC_ ),
c$$$     $               DESCB( CSRC_ ), IBSEED, 0, KP, 0, NQ, MYROW, MYCOL,
c$$$     $               NPROW, NPCOL )
c$$$      ICSEED = 300
c$$$      CALL PDMATGEN( ICTXT, 'No transpose', 'No transpose', DESCC( M_ ),
c$$$     $               DESCC( N_ ), DESCC( MB_ ), DESCC( NB_ ),
c$$$     $               MEM( IPC ), DESCC( LLD_ ), DESCC( RSRC_ ),
c$$$     $               DESCC( CSRC_ ), ICSEED, 0, MP, 0, NQ, MYROW, MYCOL,
c$$$     $               NPROW, NPCOL )
*

**********************************************************************
*     Call Level 3 PBLAS routine
**********************************************************************
*
c$$$      IF( IAM.EQ.0 ) THEN
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * )
c$$$     $         '***********************************************'
c$$$         WRITE( NOUT, FMT = * )
c$$$     $         'Example of Level 3 PBLAS routine call: (PDGEMM)'
c$$$         WRITE( NOUT, FMT = * )
c$$$     $         '***********************************************'
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * ) ' Matrix A:'
c$$$         WRITE( NOUT, FMT = * )
c$$$
c$$$         print*, "DESCA = ", DESCA
c$$$         print*, "MEM(IPA) = ", MEM(IPA:IPA+2)
c$$$         print*, "MEM(IPW) = ", MEM(IPW:IPW+2)
c$$$      END IF
      
      
c$$$      CALL PDLAPRNT( M, K, MEM( IPA ), 1, 1, DESCA, 0, 0,
c$$$     $     'A', NOUT, MEM( IPW ) )

*
c$$$      IF( IAM.EQ.0 ) THEN
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * ) ' Matrix B:'
c$$$         WRITE( NOUT, FMT = * )
c$$$      END IF
c$$$      CALL PDLAPRNT( K, N, MEM( IPB ), 1, 1, DESCB, 0, 0,
c$$$     $               'B', NOUT, MEM( IPW ) )
c$$$
c$$$      IF( IAM.EQ.0 ) THEN
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * ) ' Matrix C:'
c$$$         WRITE( NOUT, FMT = * )
c$$$      END IF
c$$$      CALL PDLAPRNT( M, N, MEM( IPC ), 1, 1, DESCC, 0, 0,
c$$$     $               'C', NOUT, MEM( IPW ) )
c$$$*
c$$$      CALL PDGEMM( 'No transpose', 'No transpose', M, N, K, ONE,
c$$$     $             MEM( IPA ), 1, 1, DESCA, MEM( IPB ), 1, 1, DESCB,
c$$$     $             ONE, MEM( IPC ), 1, 1, DESCC )
c$$$*
c$$$      IF( MYROW.EQ.0 .AND. MYCOL.EQ.0 ) THEN
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * ) ' C := C + A * B'
c$$$         WRITE( NOUT, FMT = * )
c$$$      END IF
c$$$      CALL PDLAPRNT( M, N, MEM( IPC ), 1, 1, DESCC, 0, 0,
c$$$     $               'C', NOUT, MEM( IPW ) )
c$$$*
c$$$   10 CONTINUE
c$$$*
c$$$      CALL BLACS_GRIDEXIT( ICTXT )
c$$$*
c$$$   20 CONTINUE
c$$$*
c$$$*     Print ending messages and close output file
c$$$*
c$$$      IF( IAM.EQ.0 ) THEN
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = * )
c$$$         WRITE( NOUT, FMT = 9997 )
c$$$         WRITE( NOUT, FMT = * )
c$$$         IF( NOUT.NE.6 .AND. NOUT.NE.0 )
c$$$     $      CLOSE ( NOUT )
c$$$      END IF
c$$$*
c$$$      CALL BLACS_EXIT( 0 )
c$$$*
c$$$ 9999 FORMAT( 'Bad ', A6, ' parameters: going on to next test case.' )
c$$$ 9998 FORMAT( 'Unable to perform ', A, ': need TOTMEM of at least',
c$$$     $        I11 )
c$$$ 9997 FORMAT( 'END OF TESTS.' )
*
      STOP
*
*     End of PDPBLASDRIVER
*
      END
