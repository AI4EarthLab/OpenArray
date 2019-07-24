#ifndef _INTERPLOATIO_KERNEL_HPP_
#define _INTERPLOATION_KERNEL_HPP_

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../op_define.hpp"
#include "../../MPI.hpp"
#include "../../Function.hpp"
#include "../../scalapack/pblas.h"

#include <vector>

extern "C"{
  void blacs_pinfo_(int* iam, int* nprocs);
  void blacs_setup_(int* iam, int* nprocs);
  void blacs_get_(int* icontxt, int* what, int* val);
  void blacs_gridinit_(int* icontxt, char* layout, int* nprow, int* npcol);
  void blacs_gridinfo_(int*, int*, int*, int*, int*);
  int numroc_(int* N,int* NB, int* IPROC ,int* ISRCPROC, int* NPROCS);

  void descinit_(int* DESC, int* M, int* N, int* MB, int*NB,
		 int* IRSRC, int* ICSRC, int* ICTXT, int* LLD, int* INFO);
  
  void psgemm_( char* TRANSA, char* TRANSB,
  		int * M, int * N, int * K,
  		float * ALPHA,
  		float * A, int * IA, int * JA, int * DESCA,
  		float * B, int * IB, int * JB, int * DESCB,
  		float * BETA,
  		float * C, int * IC, int * JC, int * DESCC );
 
  void pdgemm_( char* TRANSA, char* TRANSB,
  		int * M, int * N, int * K,
  		double * ALPHA,
  		double * A, int * IA, int * JA, int * DESCA,
  		double * B, int * IB, int * JB, int * DESCB,
  		double * BETA,
  		double * C, int * IC, int * JC, int * DESCC );

  void pdlaprnt_( int * m, int * n, double * A, int * ia, int * ja,
		  int * desca, int * irprnt, int * icprn, char * cmatnm,
		  int * nout,
		  double * work, int);
  
  void pslaprnt_( int * m, int * n, float * A, int * ia, int * ja,
		  int * desca, int * irprnt, int * icprn, char * cmatnm,
		  int * nout,
		  float * work, int);

  void igsum2d_(int* ICTXT, char* , char*, int*, int*, int*, int*, int*, int* );

  void igebs2d_(int* ICTXT, char*, char*, int*, int*, int*, int* );
}

namespace oa
{
	namespace kernel
	{
		template <typename TA, typename TB, typename T>
		ArrayPtr t_kernel_matrix_multiplication(vector<ArrayPtr>& ops_ap)
		{
			//row:x, column:y
			const ArrayPtr& A = ops_ap[0];
			const ArrayPtr& B = ops_ap[1];
			ArrayPtr transA, transB, ap;

            MPI_Comm comm = A->get_partition()->get_comm();
            int sw = A->get_partition()-> get_default_stencil_width();

			Shape s_A = A->shape();
			Shape s_B = B->shape();
			Shape ps = A->get_partition()->procs_shape();
           
            if (ps[2] != 1) {
                printf("z of process shape of matrix A is not 1!\n");
                return NULL;
            }

			if (s_A[1] != s_B[0]) {
                printf("y of matrix A and x of matrix B don't match!\n");
                return NULL;
            } else if (s_A[2] != s_B[2] && s_A[2] != 1 && s_B[2] != 1) {
                printf("z of two matrices don't match!\n");
                return NULL;
            }
            
			//matrix A:m*k, B:k*n
			int MB_A = (s_A[0]%ps[0]==0?(s_A[0]/ps[0]):(s_A[0]/ps[0]+1));
			int NB_A = (s_A[1]%ps[1]==0?(s_A[1]/ps[1]):(s_A[1]/ps[1]+1));
			int MB_B = (s_B[0]%ps[0]==0?(s_B[0]/ps[0]):(s_B[0]/ps[0]+1));
			int NB_B = (s_B[1]%ps[1]==0?(s_B[1]/ps[1]):(s_B[1]/ps[1]+1));
            int MB_C = MB_A;
            int NB_C = NB_B;
		
			//re-partition
			vector<int> x_A = vector<int> (ps[0], MB_A);
			vector<int> y_A = vector<int> (ps[1], NB_A);
			vector<int> z_A = vector<int> (1, s_A[2]);
			vector<int> x_B = vector<int> (ps[0], MB_B);
			vector<int> y_B = vector<int> (ps[1], NB_B);
			vector<int> z_B = vector<int> (1, s_B[2]);

            //fix process partition
            x_A[s_A[0]/MB_A] = s_A[0] % MB_A;
            for (int i = s_A[0]/MB_A+1; i < ps[0]; i++)
                x_A[i] = 0;
            y_A[s_A[1]/NB_A] = s_A[1] % NB_A;
            for (int i = s_A[1]/NB_A+1; i < ps[1]; i++)
                y_A[i] = 0;
            x_B[s_B[0]/MB_B] = s_B[0] % MB_B;
            for (int i = s_B[0]/MB_B+1; i < ps[0]; i++)
                x_B[i] = 0;
            y_B[s_B[1]/NB_B] = s_B[1] % NB_B;
            for (int i = s_B[1]/NB_B+1; i < ps[1]; i++)
                y_B[i] = 0;

			//x_A[ps[0]-1] = s_A[0] - MB_A*(ps[0] - 1);
			//y_A[ps[1]-1] = s_A[1] - NB_A*(ps[1] - 1);
			//x_B[ps[0]-1] = s_B[0] - MB_B*(ps[0] - 1);
			//y_B[ps[1]-1] = s_B[1] - NB_B*(ps[1] - 1);


			PartitionPtr pp_A = PartitionPool::global()->get(comm, x_A, y_A, z_A, sw); 
			PartitionPtr pp_B = PartitionPool::global()->get(comm, x_B, y_B, z_B, sw);
            //PartitionPtr pp_C = PartitionPool::global()->get(comm, x_A, y_B, z_B, sw);
            
            if (std::is_same<T, double>::value) {
                ap = ArrayPool::global()->get(comm, x_A, y_B, z_A[0]>z_B[0]?z_A:z_B, sw, DATA_DOUBLE);
            } else if (std::is_same<T, float>::value) {
                ap = ArrayPool::global()->get(comm, x_A, y_B, z_A[0]>z_B[0]?z_A:z_B, sw, DATA_FLOAT);
            } else if (std::is_same<T, int>::value) {
                ap = ArrayPool::global()->get(comm, x_A, y_B, z_A[0]>z_B[0]?z_A:z_B, sw, DATA_INT);
            } else {
                printf("error: wrong data type!\n");
                return NULL;
            }
			
			transA = oa::funcs::transfer(A, pp_A);
			transB = oa::funcs::transfer(B, pp_B);

			TA* buffer_transA = (TA*)transA->get_buffer();
			TB* buffer_transB = (TB*)transB->get_buffer();
			T* buffer_ap = (T*)ap->get_buffer();

			Shape local_shape_transA = transA->buffer_shape();
			Shape local_shape_transB = transB->buffer_shape();
			Shape local_shape_ap = ap->buffer_shape();

			int rank, procs;
			int ictxt, nprocs, npcol, nprow, myrow, mycol, nprow1, npcol1;
			int info;
			int ZERO_1 = 0;
			int ONE_1 = 1;
			nprow = ps[0];
			npcol = ps[1];

			blacs_pinfo_(&rank, &procs);
			blacs_get_(&ONE_1, &ZERO_1, &ictxt);
			blacs_gridinit_(&ictxt, "C", &nprow, &npcol);
			blacs_gridinfo_(&ictxt, &nprow1, &npcol1, &myrow, &mycol);

			int M = s_A[0];
			int K = s_A[1];
			int N = s_B[1];

			int descA[9], descB[9], descC[9];
			int mA = numroc_(&M, &MB_A, &myrow, &ZERO_1, &nprow);
			int nA = numroc_(&K, &NB_A, &mycol, &ZERO_1, &npcol);
			int mB = numroc_(&K, &MB_B, &myrow, &ZERO_1, &nprow);
			int nB = numroc_(&N, &NB_B, &mycol, &ZERO_1, &npcol);
			int mC = numroc_(&M, &MB_A, &myrow, &ZERO_1, &nprow);
			int nC = numroc_(&N, &NB_B, &mycol, &ZERO_1, &npcol);

			int max_1_mA = std::max(1, mA);
			int max_1_mB = std::max(1, mB);
			int max_1_mC = std::max(1, mC);
			
			descinit_(descA, &M, &K, &MB_A, &NB_A, &ZERO_1, &ZERO_1, &ictxt, &max_1_mA, &info);
			descinit_(descB, &K, &N, &MB_B, &NB_B, &ZERO_1, &ZERO_1, &ictxt, &max_1_mB, &info);
			descinit_(descC, &M, &N, &MB_C, &NB_C, &ZERO_1, &ZERO_1, &ictxt, &max_1_mC, &info);

            if (std::is_same<T, int>::value || std::is_same<T, float>::value) {
			    int i, j, k;
                float *AA = new float[mA*nA];
			    float *BB = new float[mB*nB];
			    float *CC = new float[mC*nC];
                float alpha = 1.0, beta = 0.0;

                if (s_A[2] == s_B[2]) {
                    for(k = sw; k < local_shape_transA[2] - sw; k ++)
			        {
			        	// Extract the data from matrix A and B
                        for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        		{
                                AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (float)buffer_transA[k*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        		}
			        	}
                        for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        		{
                                BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (float)buffer_transB[k*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        		}
			        	}
                        
                        // Matrix Multiplication
			        	MPI_Barrier(comm);
                        psgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
			        
                        // Store Value to Matrix C
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			        	}

			        }
                } else if (s_A[2] == 1) {
                    // AA is fixed
                    for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        {
			        	for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        	{
                            AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (float)buffer_transA[sw*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        	}
			        }
                    
                    //for z of B
                    for(k = sw; k < local_shape_transB[2] - sw; k ++)
			        {
			        	for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        		{
                                BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (float)buffer_transB[k*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        		}
			        	}
                        
			        	MPI_Barrier(comm);
                        psgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
                        
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			            }
			        }
                } else if (s_B[2] == 1) {
                    // BB is fixed
                    for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        {
			        	for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        	{
                            BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (float)buffer_transB[sw*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        	}
			        }
                    
                    //for z of A
                    for(k = sw; k < local_shape_transA[2] - sw; k ++)
			        {
			        	for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        		{
                                AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (float)buffer_transA[k*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        		}
			        	}
                        
			        	MPI_Barrier(comm);
                        psgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
                        
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			            }
			        }
                }
            } else if (std::is_same<T, double>::value) {
                int i, j, k;
                double *AA = new double[mA*nA];
			    double *BB = new double[mB*nB];
			    double *CC = new double[mC*nC];
                double alpha = 1.0, beta = 0.0;
          
                if (s_A[2] == s_B[2]) {
                    for(k = sw; k < local_shape_transA[2] - sw; k ++)
			        {
			        	for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        		{
                                AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (double)buffer_transA[k*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        		}
			        	}
			        	for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        		{
                                BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (double)buffer_transB[k*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        		}
			        	}

                        //int nout = 6;
                        //double WORK[10000];
                        //pdlaprnt_(&M, &K, AA, &ONE_1, &ONE_1, descA,
  	                    //    &ZERO_1, &ZERO_1,  "A",   &nout, WORK, 1);

                        //pdlaprnt_(&K, &N, BB, &ONE_1, &ONE_1, descB,
  	                    //    &ZERO_1, &ZERO_1,  "B",   &nout, WORK, 1);

                        
			        	MPI_Barrier(comm);
                        pdgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
			        	
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			        	}

			        }
                } else if (s_A[2] == 1) {
                    // AA is fixed
                    for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        {
			        	for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        	{
                            AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (double)buffer_transA[sw*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        	}
			        }
                    
                    //for z of B
                    for(k = sw; k < local_shape_transB[2] - sw; k ++)
			        {
			        	for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        		{
                                BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (double)buffer_transB[k*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        		}
			        	}
                        
			        	MPI_Barrier(comm);
                        pdgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
                        
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			            }
			        }
                } else if (s_B[2] == 1) {
                    // BB is fixed
                    for(j = sw; j < local_shape_transB[1] - sw; j ++)
			        {
			        	for(i = sw; i < local_shape_transB[0] - sw; i ++)
			        	{
                            BB[(j-sw)*(local_shape_transB[0]-2*sw) + (i-sw)] = (double)buffer_transB[sw*local_shape_transB[1]*local_shape_transB[0] + j*local_shape_transB[0] + i];
			        	}
			        }
                    
                    //for z of A
                    for(k = sw; k < local_shape_transA[2] - sw; k ++)
			        {
			        	for(j = sw; j < local_shape_transA[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_transA[0] - sw; i ++)
			        		{
                                AA[(j-sw)*(local_shape_transA[0]-2*sw) + (i-sw)] = (double)buffer_transA[k*local_shape_transA[1]*local_shape_transA[0] + j*local_shape_transA[0] + i];
			        		}
			        	}
                        
			        	MPI_Barrier(comm);
                        pdgemm_("N", "N", &M, &N, &K, 
			        	    		&alpha, 
			        	    		AA, &ONE_1, &ONE_1, descA, 
			        	    		BB, &ONE_1, &ONE_1, descB, 
			        	    		&beta, 
			        	    		CC, &ONE_1, &ONE_1, descC);
                        MPI_Barrier(comm);
                        
                        for(j = sw; j < local_shape_ap[1] - sw; j ++)
			        	{
			        		for(i = sw; i < local_shape_ap[0] - sw; i ++)
			        		{
			        			buffer_ap[k*local_shape_ap[1]*local_shape_ap[0] + j*local_shape_ap[0] + i] = (T)CC[(j-sw)*(local_shape_ap[0]-2*sw) + (i-sw)];
			        		}
			            }
			        }
                }
            } else {
                printf("data type error!\n");
                return ap;
            }
            
			return ap;
		}
		
		ArrayPtr kernel_mat_mult(vector<ArrayPtr> &ops_ap);
	}
}


#endif
