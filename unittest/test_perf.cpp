#include <iostream>
#include "../common.hpp"
#include "../Array.hpp"
#include "../Function.hpp"
#include "../Node.hpp"
#include "../Operator.hpp"
#include "mpi.h"

extern "C"{
  void tic(const char* s);
  void toc(const char* s);
  void show_time(const char* s);
  void show_all();
}

void test_sample_math(){
  int m = 512, n = 512, p = 512;

  MPI_Comm comm = MPI_COMM_WORLD;
  
  ArrayPtr A =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);
  ArrayPtr B =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);
  ArrayPtr C =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);

  ///:for op in ['DIVD', 'MINUS', 'MULT', 'PLUS']
  tic("${op}$");    
  for(int i = 0; i < 10; ++i){
    NodePtr X = oa::ops::new_node(A);
    NodePtr Y = oa::ops::new_node(B);

    NodePtr Z = oa::ops::new_node(TYPE_${op}$, X, Y);
    ArrayPtr T = oa::ops::eval(Z);
  }
  toc("${op}$");  
  ///:endfor
}

int main(int argc, char** argv){

  MPI_Init(&argc, &argv);

  test_sample_math();
  
  show_all();
  
  MPI_Finalize();
  return 0;
}

