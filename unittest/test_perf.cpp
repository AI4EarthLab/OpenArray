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
  int m = 512, n = 512, p = 256;

  MPI_Comm comm = MPI_COMM_WORLD;
  
  ArrayPtr A =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);
  ArrayPtr B =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);
  ArrayPtr C =
    oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<double>::type);


  for(int i = 0; i < 10; ++i){
    tic("A");    
    NodePtr X = oa::ops::new_node(A);
    NodePtr Y = oa::ops::new_node(B);

    NodePtr Z = oa::ops::new_node(TYPE_PLUS, X, Y);
    ArrayPtr T = oa::ops::eval(Z);
    toc("A");
  }

  ///:for dt in ['float', 'double']
  {
    ArrayPtr A =
      oa::funcs::seqs(comm,{m, n, p}, 1, oa::utils::dtype<${dt}$>::type);
    ArrayPtr B =
      oa::funcs::seqs(comm,{m, n, p}, 1, oa::utils::dtype<${dt}$>::type);
    ArrayPtr C =
      oa::funcs::ones(comm,{m, n, p}, 1, oa::utils::dtype<${dt}$>::type);

    ///:for op in ['PLUS', 'MINUS', 'MULT', 'DIVD']
    tic("${op}$(${dt}$)");    
    for(int i = 0; i < 10; ++i){
      NodePtr X = oa::ops::new_node(A);
      NodePtr Y = oa::ops::new_node(B);

      NodePtr Z = oa::ops::new_node(TYPE_${op}$, X, Y);
      ArrayPtr T = oa::ops::eval(Z);
      //printf("%p\n", T->get_buffer());
      //T->display("T : ");
    }
    toc("${op}$(${dt}$)");
    ///:endfor
  }
  ///:endfor

}

int main(int argc, char** argv){

  MPI_Init(&argc, &argv);

  test_sample_math();
  
  show_all();
  
  MPI_Finalize();
  return 0;
}
