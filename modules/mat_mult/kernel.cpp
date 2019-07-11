#include "kernel.hpp"
#include <stdio.h>

namespace oa
{
	namespace kernel
	{	
		ArrayPtr kernel_mat_mult(vector<ArrayPtr> &ops_ap)
		{
			ArrayPtr A = ops_ap[0];
			ArrayPtr B = ops_ap[1];

			int A_dt = A->get_data_type();
			int B_dt = B->get_data_type();

            switch(A_dt) {
                case DATA_DOUBLE:
                    switch(B_dt) {
                        case DATA_DOUBLE:
                            return t_kernel_matrix_multiplication<double,double,double>(ops_ap);
                        case DATA_FLOAT:
                            return t_kernel_matrix_multiplication<double,float,double>(ops_ap);
                        case DATA_INT:
                            return t_kernel_matrix_multiplication<double,int,double>(ops_ap);
                    }
                case DATA_FLOAT:
                    switch(B_dt) {
                        case DATA_DOUBLE:
                            return t_kernel_matrix_multiplication<float,double,double>(ops_ap);
                        case DATA_FLOAT:
                            return t_kernel_matrix_multiplication<float,float,float>(ops_ap);
                        case DATA_INT:
                            return t_kernel_matrix_multiplication<float,int,float>(ops_ap);
                    }
                case DATA_INT:
                    switch(B_dt) {
                        case DATA_DOUBLE:
                            return t_kernel_matrix_multiplication<int,double,double>(ops_ap);
                        case DATA_FLOAT:
                            return t_kernel_matrix_multiplication<int,float,float>(ops_ap);
                        case DATA_INT:
                            return t_kernel_matrix_multiplication<int,int,int>(ops_ap);
                    }
            }

            printf("Error: Wrong data type!\n");
            return NULL;

            //if (A_dt == DATA_DOUBLE || B_dt == DATA_DOUBLE) {
            //    printf("enter double\n");
            //    return t_kernel_matrix_multiplication<double>(ops_ap);
            //} else if (A_dt == DATA_FLOAT || B_dt == DATA_FLOAT) {
            //    printf("enter float\n");
            //    return t_kernel_matrix_multiplication<float>(ops_ap);
            //} else if (A_dt == DATA_INT && B_dt == DATA_INT) {
            //    printf("enter int\n");
            //    return t_kernel_matrix_multiplication<int>(ops_ap);
            //} else {
            //    printf("Unsupported data types! \n");
            //    return NULL;
            //}
		}
	}
}
