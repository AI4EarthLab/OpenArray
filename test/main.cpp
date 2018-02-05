#include <iostream>
#include <cstdio>
#include <string>
#include <mpi.h>
#include "test.hpp"
#include "../MPI.hpp"

using namespace std;

int main(int argc, char** argv) {
  
  // MPI_Init(NULL, NULL);
  oa::MPI::global()->c_init(MPI_COMM_WORLD, 0, NULL);

  int m = stol(argv[1]);
  int n = stol(argv[2]);
  int p = stol(argv[3]);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  if (world_rank == 0) {
  //  test_Range();
  //  test_Box();
  //  test_Partition();
  }
  
  //test_Pool();
  //test_Array();

  // mpirun -n 6 for test_transfer 
  //test_transfer();
  test_update_ghost();

  //test_operator();
  //test_io();

  //test_operator();
  //test_write_graph();
  //test_force_eval();
  //test_fusion_kernel();
  //test_c_interface();
  //test_logic_operator();
  //test_math_operator();
  //test_gen_kernel_JIT();
  //test_min_max();
  //test_eval();
  //test_csum();
  //test_sum();
  //test_sub();
  //test_set();
  //test_rep();
  //test_l2g();
  //test_g2l();
  //test_set_l2g();
  //test_set_g2l();
  //test_set_with_mask();
  //test_fusion_operator();
  //test_op();

  // tic("3d");
  // for (int i = 0; i < 10; i++)
  // test_fusion_op_3d(m, n, p, 1);
  // toc("3d");
  // tic("2d");
  // test_fusion_op_2d(m, n, p);
  // toc("2d");

  // show_all();


  //test_pseudo_3d();
  //test_rand();
  //test_bitset();
  //test_operator_with_grid();
  
  // tic("cache");
  // test_cache(m, n, p);
  // toc("cache");

  // show_all();

  //test_gen_kernel_JIT_with_op(m, n, p);

  if (world_rank == 0) std::cout<<"Finished."<<std::endl;
  // !clear_cache();
  
  MPI_Finalize();
  // clear_cache();
  return 0;
}
