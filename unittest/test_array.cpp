

#include <armadillo>
#include "../Array.hpp"
#include "../Function.hpp"
#include "../NodePool.hpp"
#include "../Operator.hpp"
#include "gtest/gtest.h"


namespace gt=::testing;

#define MRange gt::Range(3, 4)
#define NRange gt::Range(3, 4)
#define PRange gt::Range(3, 4)

class MPITest : public gt::TestWithParam
<std::tr1::tuple<MPI_Comm, int, int, int> > {
protected:
  virtual void SetUp() {
    printf("SetUP MPITest!!!\n");
    comm = std::tr1::get<0>(GetParam());

    m  = std::tr1::get<1>(GetParam());
    n  = std::tr1::get<2>(GetParam());
    p  = std::tr1::get<3>(GetParam());

    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
  }

  virtual void TearDown() {
    
  }

  MPI_Comm comm;
  int size;
  int rank;

  int m;
  int n;
  int p;
};

template<class T>
arma::Cube<T> make_seqs(int m, int n, int p){
  arma::Cube<T> v1(m, n, p);

  int cnt = 0;
  for(int k = 0; k < p; ++k){
    for(int j = 0; j < n; ++j){
      for(int i = 0; i < m; ++i){
	v1(i, j, k) = cnt;
	cnt++;
      }
    }
  }

  return v1;
}

namespace{

  TEST_P(MPITest, ArrayCreation){

    ///:for t in ['int', 'float', 'double']
    {
      ArrayPtr A1 = oa::funcs::to_rank0(oa::funcs::seqs(comm, {m, n, p}, 1,
							oa::utils::dtype<${t}$>::type));
      arma::Cube<${t}$> B1 = make_seqs<${t}$>(m, n, p);

      ArrayPtr A2 =
	oa::funcs::to_rank0(oa::funcs::consts<${t}$>(comm, {m, n, p},
						     ${t}$(2),
						     oa::utils::dtype<${t}$>::type));
      arma::Cube<${t}$> B2(m, n, p);
      B2.fill(${t}$(2));
      
      if(rank == 0){
	EXPECT_TRUE(oa::funcs::is_equal(A1, B1));
	EXPECT_TRUE(oa::funcs::is_equal(A2, B2));
      }
    }
    ///:endfor
  }


  TEST_P(MPITest, BasicMath){

    ///:for t in ['int', 'float', 'double']
    {

      typedef arma::Cube<${t}$> cube_t;
      
      DataType dt  = oa::utils::dtype<${t}$>::type;
      NodePtr N1 = oa::ops::new_node(oa::funcs::seqs(comm, {m, n, p}, 1, dt));
      NodePtr N2 = oa::ops::new_node(oa::funcs::ones(comm, {m, n, p}, 1, dt));
      NodePtr N3 = oa::ops::new_node(oa::funcs::consts(comm, {m, n, p}, ${t}$(3.0), 1));

      {
	NodePtr F = oa::ops::new_node(TYPE_MINUS, N1, N2);
	oa::ops::eval(F)->display("TYPE_MINUS(${t}$):");
      }
      
      // oa::ops::eval(F)->display("TYPE_PLUS :");
      // F = oa::ops::new_node(TYPE_MINUS, N1, N2);
      // X1 = oa::funcs::to_rank0(oa::ops::eval(F));
      //oa::ops::eval(F)->display("TYPE_MINUS :");
      
      // cube_t B1 = make_seqs<${t}$>(m, n, p);
      // cube_t B2 = arma::ones<cube_t>(m, n, p);	

      ///:for o in [['+','PLUS'], ['-', 'MINUS'], ['%','MULT'], ['/', 'DIVD']]
      {
	// NodePtr F = oa::ops::new_node(TYPE_${o[1]}$, N1, N2);

	// oa::ops::eval(F)->display("TYPE_${o[1]}$1 :");
			    
	// ArrayPtr X1 = oa::funcs::to_rank0(oa::ops::eval(F));
	
	// if(rank == 0){
	//   X1->display("TYPE_${o[1]}$ :");
	// cube_t Y1 = B1 ${o[0]}$ B2;
	// std::cout<< "B1:" <<std::endl<< B1 << std::endl;
	// std::cout<< "B2:" <<std::endl<< B2 << std::endl;	  
	// std::cout<< "B1 ${o[0]}$ B2:" <<std::endl<< B1 ${o[0]}$ B2 << std::endl;
	  
	//EXPECT_TRUE(oa::funcs::is_equal(X1, Y1));
	//}
      }
      ///:endfor
		
      // // ((A+B)-(C*D))/E
      // NodePtr A = oa::ops::new_node(ap1);
      // NodePtr B = oa::ops::new_node(ap2);
      // NodePtr C = oa::ops::new_node(ap3);

      // NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
      // NodePtr G = oa::ops::new_node(TYPE_PLUS, F, C);
      // ArrayPtr ans = oa::funcs::to_rank0(oa::ops::eval(G));

      // if(rank == 0){
      // 	// arma::Cube<${t}$> A1(m, n, p);
      // 	// arma::Cube<${t}$> A2(m, n, p);	
      // 	arma::Cube<${t}$> ans1 = make_seqs<${t}$>(m, n, p)
      // 			       + arma::ones<arma::Cube<${t}$> >(m, n, p) * 4;

      // 	// std::cout<<ans1<<std::endl;
      // 	// ans->display("A+B+C");
	
      // 	EXPECT_TRUE(oa::funcs::is_equal(ans, ans1));
	
      // }
      //ans->display("A+B+C");
    }
    ///:endfor
  }

  INSTANTIATE_TEST_CASE_P(OpenArray, MPITest,
			  gt::Combine(gt::Values(MPI_COMM_WORLD),
				      MRange, NRange, PRange));

  
}
