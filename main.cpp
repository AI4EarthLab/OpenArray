#include <iostream>
#include <cstdio>
#include <string>
#include <mpi.h>
#include "./test/test.hpp"

using namespace std;

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
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
  //test_transfer();
  //test_update_ghost();
  test_operator();

  MPI_Finalize();
  return 0;
}
