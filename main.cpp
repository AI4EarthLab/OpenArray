#include<iostream>
#include<cstdio>
#include<string>
#include<mpi.h>
#include<assert.h>
#include "Range.hpp"

using namespace std;

void test_range() {
    Range A(1, 2);
    Range B;
    A.display("A");
    B.display();
    assert(A.equal(1, 2));
    assert(!A.equal(B));
    cout<<A.size()<<endl;
    B.shift(2);
    B.display("B");
}

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
        test_range();
    }
   
    MPI_Finalize();
    return 0;
}
