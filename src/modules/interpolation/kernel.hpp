#ifndef _INTERPLOATION_KERNEL_HPP_
#define _INTERPLOATION_KERNEL_HPP_

#include "../../NodePool.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../op_define.hpp"
#include "../../MPI.hpp"

#include <vector>

namespace oa
{
	namespace kernel
	{
		template <typename T>
		struct data_to_trans
		{
			T data;
			int x;
			int y;
			int z;
			//int cnt;
		};

		template <typename T>
		ArrayPtr t_kernel_interpolation(vector<ArrayPtr>& ops_ap)
		{
			const ArrayPtr& A = ops_ap[0];
			const ArrayPtr& d = ops_ap[1];

			int* dim = (int*) d->get_buffer();
			int x, y, z;
			x = dim[0];
			y = dim[1];
			z = dim[2];
					
			ArrayPtr ap, ap1;
			Shape s_A = A->shape();
			Shape ps_A = A->get_partition()->procs_shape();
			//Shape ls = A->local_shape();

			//printf("A->local_shape=[%d, %d, %d]\n", ls[0], ls[1], ls[2]);
			//printf("A->global_shape=[%d, %d, %d]\n", s_A[0], s_A[1], s_A[2]);

			int sw = A->get_partition()->get_stencil_width();
			
			ap1 = ArrayPool::global()->get(MPI::global()->comm(), 
					{{s_A[0], s_A[1], s_A[2]}}, sw, 
					A->get_data_type());

			ap = ArrayPool::global()->get(MPI::global()->comm(), 
					{{s_A[0]/x, s_A[1]/y, s_A[2]/z}}, sw, 
					A->get_data_type());

			Shape s_ap = ap->shape();
			Shape ps_ap = ap->get_partition()->procs_shape();
			//Shape ls_ap = ap->local_shape();

			//printf("ap->local_shape=[%d, %d, %d]\n", ls_ap[0], ls_ap[1], ls_ap[2]);
			//printf("ap->global_shape=[%d, %d, %d]\n", s_ap[0], s_ap[1], s_ap[2]);

			//int xs, xe, ys, ye, zs, ze;
			//ap->get_local_box().get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
			//printf("ap->box_shape=[[%d, %d), [%d, %d), [%d, %d)]\n", xs, xe, ys, ye, zs, ze);
			//A->display("A in t_kernel_interpolation");	
			T* buffer_A = (T*)A->get_buffer();
			T* buffer_ap = (T*)ap->get_buffer();
			T* buffer_ap1 = (T*)ap1->get_buffer();
		
			Shape shape_A = A->buffer_shape();
			Shape shape_ap = ap->buffer_shape();

			Shape global_shape_A = A->get_partition()->shape();
			Shape global_shape_ap = ap->get_partition()->shape();

			int  global_cnt_x_A = global_shape_A[0]/x;
			int  global_cnt_y_A = global_shape_A[0]/x;
			int  global_cnt_z_A = global_shape_A[0]/x;

			int cnt_x = shape_ap[0];
			int cnt_y = shape_ap[1];
			int cnt_z = shape_ap[2];
						
			int cnt_x_A = shape_A[0];
			int cnt_y_A = shape_A[1];
			int cnt_z_A = shape_A[2];

			Box b;
			vector<int> v_int(3, 0);
			int rank;
			int xs, xe, ys, ye, zs, ze;
			b = ap->get_local_box();
			Box b_A = A->get_local_box();
			rank = A->get_partition()->rank();
			v_int = A->get_partition()->get_procs_3d(rank);
			
			//printf("rank=%d, ap->buffer_size=[%d, %d, %d]\n", rank, shape_ap[0], shape_ap[1], shape_ap[2]);
			
			//printf("rank=%d\n", rank);
			//printf("v_int=[%d, %d, %d]\n", v_int[0], v_int[1], v_int[2]);
			//printf("rank=%d, b=[%d, %d, %d, %d, %d, %d]\n", rank, b.xs(), b.xe(), b.ys(), b.ye(), b.zs(), b.ze());
			//std::cout<<"cnt_x_A="<<cnt_x_A<<std::endl;
			//std::cout<<"cnt_y_A="<<cnt_y_A<<std::endl;
			//std::cout<<"cnt_z_A="<<cnt_z_A<<std::endl;

			oa::funcs::set_ghost_zeros(A);
			oa::funcs::update_ghost(A);	
			//bool lbx=false, lby=false, lbz=false, rbx=false, rby=false, rbz=false;
			int sc = 0;
			int bx=0, by=0, bz=0;	
			
			//glb=global left boundary  //grb=global right boundary
			bool glbx, glby, glbz, grbx, grby, grbz;
			glbx=glby=glbz=grbx=grby=grbz=false;
	
			/*for(int k = 0; k < cnt_z_A ; k++)
				for(int j = 0; j < cnt_y_A ; j++)
					for(int i = 0; i < cnt_x_A ; i++)
						std::cout<<"buffer_A="<<buffer_A[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i]<<std::endl;
			*/

			if(b_A.xs() == 0)
				glbx = true;
			if(b_A.xe() == (s_A[0]))
				grbx = true;
			if(b_A.ys() == 0)
				glby = true;
			if(b_A.ye() == (s_A[1]))
				grby = true;
			if(b_A.zs() == 0)
				glbz = true;
			if(b_A.ze() == (s_A[2]))
				grbz = true;

			if(!A->has_local_data())
				return ap;

			for(int k = sw; k < cnt_z_A - sw; k++)
			{
				for(int j = sw; j < cnt_y_A - sw; j++)
				{
					for(int i = sw; i < cnt_x_A - sw; i++)
					{
						if(k == sw)
						{
							if(glbz==true) 
							{
								bz ++ ;
							}
						}
						if(k == cnt_z_A - sw - 1) 
						{
							if(grbz == true)
							{
								bz ++;
							}
						}
						if(j == sw)
						{
							if(glby == true) 
							{
								by ++;
							}
						}
						if(j == cnt_y_A - sw - 1) 
						{
							if(grby == true) 
							{
								by ++;
							}
						}
						if(i == sw)
						{
							if(glbx == true) 
							{
								bx ++;
							}
						}
						if(i == cnt_x_A - sw - 1) 
						{
							if(grbx == true) 
							{
								bx ++;	
							}
						}


						buffer_ap1[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i] = 
							//same layer
							buffer_A[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i] +
							
							buffer_A[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i-1] +
							buffer_A[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i+1] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i-1] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i+1] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i-1] +
							buffer_A[k*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i+1] +

							//upper layer
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i-1] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i+1] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i-1] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i+1] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i-1] +
							buffer_A[(k+1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i+1] +

							//lower layer
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i-1] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + j*cnt_x_A + i+1] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i-1] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j-1)*cnt_x_A + i+1] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i-1] +
							buffer_A[(k-1)*cnt_x_A*cnt_y_A + (j+1)*cnt_x_A + i+1] ;
							
						if(bx == 0)
						{
							if(by == 0)
							{
								if(bz == 0)
								{
									sc = 27;
								}
								else if(bz == 1)
								{
									sc = 18;
								}
								else if(bz == 2)
								{
									sc = 9;
								}
							}
							else if(by == 1)
							{
								if(bz == 0)
								{
									sc = 18;
								}
								else if(bz == 1)
								{
									sc = 12;
								}
								else if(bz == 2)
								{
									sc = 6;
								}
							}
							else if(by == 2)
							{
								if(bz == 0)
								{
									sc = 9;
								}
								else if(bz == 1)
								{
									sc = 12;
								}
								else if(bz == 2)
								{
									sc = 3;
								}
							}
						}
						else if(bx == 1)
						{
							if(by == 0)
							{
								if(bz == 0)
								{
									sc = 18;
								}
								else if(bz == 1)
								{
									sc = 12;
								}
								else if(bz == 2)
								{
									sc = 6;
								}
							}
							else if(by == 1)
							{
								if(bz == 0)
								{
									sc = 12;
								}
								else if(bz == 1)
								{
									sc = 8;
								}
								else if(bz == 2)
								{
									sc = 2;
								}
							}
							else if(by == 2)
							{
								if(bz == 0)
								{
									sc = 6;
								}
								else if(bz == 1)
								{
									sc = 4;
								}
								else if(bz == 2)
								{
									sc = 2;
								}
							}
						}
						else if(bx == 2)
						{
							if(by == 0)
							{
								if(bz == 0)
								{
									sc = 9;
								}
								else if(bz == 1)
								{
									sc = 6;
								}
								else if(bz = 2)
								{
									sc = 3;
								}
							}
							else if(by == 1)
							{
								if(bz == 0)
								{
									sc = 6;
								}
								else if(bz == 1)
								{
									sc = 4;
								}
								else if(bz == 2)
								{
									sc = 2;
								}
							}
							else if(by == 2)
							{
								if(bz == 0)
								{
									sc = 3;
								}
								else if(bz == 1)
								{
									sc = 2;
								}
								else if(bz == 2)
								{
									sc = 1;
								}
							}
						}
						//printf("rank=%d, bx=%d, by=%d, bz=%d\n", rank, bx, by, bz);
						
						//printf("[%d, %d, %d, %d, %d, %d], [%d, %d, %d], sc=%d\n", glbx, grbx, glby, grby, glbz ,grbz, bx, by, bz, sc);	
						bx = by = bz = 0;
						//printf("sc=%d\n", sc);	
						buffer_ap1[k*cnt_x_A*cnt_y_A + j*cnt_x_A + i] /= sc;
						//printf("buffer_ap1=%fi\n", buffer_ap1[k*cnt_x_A*cnt_y_A + j*cnt_x_A + x*i]);

					}
				}
			}	

			//printf("rank=%d, glbx=%d, grbx=%d, glby=%d, grby=%d, glbz=%d, grbz=%d", rank, glbx, grbx, glby, grby, glbz, grbz);
			glbx=glby=glbz=grbx=grby=grbz=false;

			int data_cnt = 0;	

			data_to_trans<T> v[(cnt_x-2*sw)*(cnt_y-2*sw)*(cnt_z-2*sw)];
			MPI_Request reqs[ps_A[0]*ps_A[1]*ps_A[2]];
			MPI_Request reqs_send[ps_A[0]*ps_A[1]*ps_A[2]];
			int reqs_cnt = 0; 
			int reqs_send_cnt = 0;

			//int size_dt = sizeof(data_to_trans<double>);
			//printf("sizeof(data_to_trans)=%d", size_dt);

			int glx_A;
			int gly_A;
			int glz_A;
			if(ap->has_local_data())
			{
				printf("rank=%d, has_local_data\n", rank);
				printf("rank=%d, b_A=[%d, %d, %d, %d, %d, %d]\n", rank, b_A.xs(), b_A.xe(), b_A.ys(), b_A.ye(), b_A.zs(), b_A.ze());
				for(int k = 0; k < global_shape_ap[2]; k++)				//自己需要数据
				{
					for(int j = 0; j < global_shape_ap[1]; j++)
					{
						for(int i = 0; i < global_shape_ap[0]; i++)
						{	
							Box temp_box_ap = Box(i, i+1, j, j+1, k, k+1);
							if(!temp_box_ap.is_inside(b)) continue;
							glx_A = (i+1)*x-1;
							gly_A = (j+1)*y-1;
							glz_A = (k+1)*z-1;
							Box temp_box_A = Box(glx_A, glx_A+1, gly_A, gly_A+1, glz_A, glz_A+1);
							printf("rank=%d, [i, j, k]=[%d, %d, %d], [glx, gly, glz]=[%d, %d, %d]\n", rank, i ,j, k, glx_A, gly_A, glz_A);
							if(temp_box_A.is_inside(b_A)) 			//需要的数据在本进程
							{
								printf("++++++++++++++++++++++\n");

								int x_A = glx_A - b_A.xs() + sw;
								int y_A = gly_A - b_A.ys() + sw;
								int z_A = glz_A - b_A.zs() + sw; 
								int x_ap = i - b.xs() + sw;
								int y_ap = j - b.ys() + sw;
								int z_ap = k - b.zs() + sw;
								printf("rank=%d, [x_ap, y_ap, x_ap]=[%d, %d, %d], [x_A, y_A, z_A]=[%d, %d, %d] \n", rank, x_ap, y_ap, z_ap, x_A, y_A, z_A);
								buffer_ap[z_ap*cnt_x*cnt_y + y_ap*cnt_x + x_ap] = buffer_ap1[z_A*cnt_x_A*cnt_y_A + y_A*cnt_x_A + x_A];
								printf("rank=%d, ,buffer_ap[%d]=%f, buffer_ap1[%d]=%f", 
										rank, z_ap*cnt_x*cnt_y + y_ap*cnt_x + x_ap, 
										buffer_ap[z_ap*cnt_x*cnt_y + y_ap*cnt_x + x_ap],
										z_A*cnt_x_A*cnt_y_A + y_A*cnt_x_A + x, 
										buffer_ap1[z_A*cnt_x_A*cnt_y_A + y_A*cnt_x_A + x]);
							}
							else														//需要的数据不在本进程
							{
								for(int npz = 0; npz < ps_A[2]; npz ++)
								{
									for(int npy = 0; npy < ps_A[1]; npy ++)
									{
										for(int npx = 0; npx < ps_A[0]; npx ++)
										{
											Box box = A->get_partition()->get_local_box({npx, npy, npz});
											//printf("rank=%d, box=[%d, %d, %d, %d, %d, %d]\n", rank, box.xs(), box.xe(), box.ys(), box.ye(), box.zs(), box.ze());
											if(box.size() <= 0) 
												continue;
											if(!temp_box_A.is_inside(box)) 
												//printf("assert is_inside() failed!\n");
												continue;
											//printf("rank=%d, MPI_Irecv from !\n", rank, );
											int target_rank = A->get_partition()->get_procs_rank({npx, npy, npz});
											printf("rank=%d, MPI_Irecv from %d!\n", rank, target_rank);

											MPI_Irecv(&v[reqs_cnt], sizeof(data_to_trans<T>),
													MPI_BYTE, 
													target_rank, 100, 
													A->get_partition()->get_comm(),
													&reqs[reqs_cnt]
													);
											reqs_cnt++;
											//printf("++++++++++++++++++++++++++++++++++++++++++++++done++++++++++++++++++++++++++++++++++++++++\n");
										}
									}
								}
							}
						}
					}
				}

				for(int k = 0; k < global_shape_A[2]; k++)				//有别的进程的数据
				{
					for(int j = 0; j < global_shape_A[1]; j++)
					{
						for(int i = 0; i < global_shape_A[0]; i++)
						{
							if((i+1)%x==0 && (j+1)%y==0 && (k+1)%z==0)
							{
								Box box_temp_ap = Box((i+1)/x-1, (i+1)/x, (j+1)/y-1, (j+1)/y, (k+1)/z-1, (k+1)/z);
								Box box_temp_A = Box(i, i+1, j, j+1, k, k+1);
								printf("test, rank=%d, box_temp_A=[%d, %d, %d]\n", rank, i ,j, k);
								if(!box_temp_A.is_inside(b_A)) continue;
								for(int npz = 0; npz < ps_ap[2]; npz++)
								{
									for(int npy = 0; npy < ps_ap[1]; npy++)
									{
										for(int npx = 0; npx < ps_ap[0]; npx++)
										{
											Box box = ap->get_partition()->get_local_box({npx, npy, npz});
											if(box.size() <= 0) continue;
											if(box_temp_ap.is_inside(box) && !box_temp_ap.is_inside(b))
											{
												int target_rank = ap->get_partition()->get_procs_rank({npx, npy, npz});
												printf("rank=%d, process has others data!, target_rank=%d\n", rank, target_rank);
												printf("rank=%d, target_rank=%d, box_temp_ap=[%d, %d, %d]\n", 
														rank, target_rank, box_temp_ap.xs(), box_temp_ap.ys(), box_temp_ap.zs());										
												data_to_trans<T> dt;
												int temp_cnt_x = i - b_A.xs() + sw;
												int temp_cnt_y = j - b_A.ys() + sw;
												int temp_cnt_z = k - b_A.zs() + sw;
												dt.data = buffer_ap1[temp_cnt_z*cnt_x_A*cnt_y_A + temp_cnt_y*cnt_x_A + temp_cnt_x];
												dt.x = (i+1)/x-1;
												dt.y = (j+1)/y-1;
												dt.z= (k+1)/z-1;
												MPI_Isend(&dt, sizeof(data_to_trans<T>),
														MPI_BYTE,
														target_rank, 100,
														A->get_partition()->get_comm(),
														&reqs_send[reqs_send_cnt]
													);
												reqs_send_cnt++;
											}
										}
									}
								}
							}
						}
					}
				}
			}

			else													//A有数据而ap无数据，需判断A中是否有其他进程所需的数据
			{
				printf("rank=%d, ap->has_local_data() is false!\n", rank);
				for(int k = 0; k < global_shape_A[2]; k++)
				{
					for(int j = 0; j < global_shape_A[1]; j++)
					{
						for(int i = 0; i < global_shape_A[0]; i++)
						{ 
							//printf("rank=%d, [i, j, k] = [%d, %d, %d]\n", rank, i, j, k);
							if((i+1)%x==0 && (j+1)%y==0 && (k+1)%z==0)
							{
								//printf("+++++++++++++++++++++++++if+++++++++++++++++++++++++++\n");
								Box box_temp_ap = Box((i+1)/x-1, (i+1)/x, (j+1)/y-1, (j+1)/y, (k+1)/z-1, (k+1)/z);
								Box box_temp_A = Box(i, i+1, j, j+1, k, k+1);
								printf("rank=%d, box_temp_ap=[%d, %d, %d]\n", rank, (i+1)/x-1, (j+1)/y-1, (k+1)/z-1);
								if(!box_temp_A.is_inside(b_A)) continue;
								//printf("rank=%d, box_temp_A=[%d, %d, %d]\n", rank, i, j, k);
								printf("rank=%d, global_shape_A=[%d, %d, %d]\n", rank, global_shape_A[0], global_shape_A[1], global_shape_A[2]);
								//printf("true, rank=%d, [i, j, k] = [%d, %d, %d]\n", rank, i, j, k);

								for(int npz = 0; npz < ps_ap[2]; npz++)
								{
									for(int npy = 0; npy < ps_ap[1]; npy++)
									{
										for(int npx = 0; npx < ps_ap[0]; npx++)
										{
											Box box = ap->get_partition()->get_local_box({npx, npy, npz});
											if(box.size() <= 0) continue;
											if(!box_temp_ap.is_inside(box)) continue;
											int target_rank = ap->get_partition()->get_procs_rank({npx, npy, npz});

											printf("true, target_rank=%d, [i, j, k] = [%d, %d, %d]\n", target_rank, i, j, k);
											printf("rank=%d, target_rank=%d, MPI_Send\n", rank, target_rank);
														//int qtarget_rank = ap->get_partition()->get_procs_rank({npx, npy, npz});
											
											data_to_trans<T> dt;
											int temp_cnt_x = i - b_A.xs() + sw;
											int temp_cnt_y = j - b_A.ys() + sw;
											int temp_cnt_z = k - b_A.zs() + sw;
											dt.data = buffer_ap1[temp_cnt_z*cnt_x_A*cnt_y_A + temp_cnt_y*cnt_x_A + temp_cnt_x];
											dt.x = (i+1)/x-1;
											dt.y = (j+1)/y-1;
											dt.z= (k+1)/z-1;
											MPI_Send(&dt, sizeof(data_to_trans<T>),
													MPI_BYTE,
													target_rank, 100,
													A->get_partition()->get_comm()
												);
										}
									}
								}
							}
						}
					}
				}
			}
			//MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);

			if(ap->has_local_data())
			{
				MPI_Waitall(reqs_send_cnt, &reqs_send[0], MPI_STATUSES_IGNORE);
				MPI_Waitall(reqs_cnt, &reqs[0], MPI_STATUSES_IGNORE);

				for(int t = 0; t < reqs_cnt; t ++)
				{
					int i = v[t].x - b.xs()  +sw;
					int j = v[t].y - b.ys() + sw;
					int k = v[t].z - b.zs() + sw;
					buffer_ap[k*cnt_x*cnt_y + j*cnt_x + i] = v[t].data;
				}
			}
			return ap;
		}

		ArrayPtr kernel_interpolation(vector<ArrayPtr> &ops_ap);
	}
}


#endif
