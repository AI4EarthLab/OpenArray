

#include <armadillo/armadillo>
#include "../Array.hpp"
#include "../Function.hpp"
#include "../NodePool.hpp"
#include "../Operator.hpp"
#include "../IO.hpp"
#include "gtest/gtest.h"
#include <boost/filesystem.hpp>

namespace gt=::testing;

#define MRange gt::Range(1, 4)
#define NRange gt::Range(1, 4)
#define PRange gt::Range(1, 4)

class MPITest : public gt::TestWithParam
<std::tr1::tuple<MPI_Comm, int, int, int> > {
protected:
  virtual void SetUp() {

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

///:set dtypes = ['int', 'float', 'double']
// ///:set dtypes = ['int']

using namespace oa::funcs;

namespace{

  TEST(Array, Basic){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    ArrayPtr A1 = ones(MPI_COMM_SELF, {10,10,10}, 1, DATA_INT);
    if(rank == 0){
      EXPECT_FALSE(A1->is_scalar());
      EXPECT_FALSE(A1->is_seqs_scalar());
      EXPECT_TRUE(A1->is_seqs());
      EXPECT_EQ(A1->shape(), Shape({10,10,10}));
    }

    ArrayPtr A2 = ones(MPI_COMM_WORLD, {1,1,1}, 1, DATA_INT);
    if(rank == 0){
      EXPECT_TRUE(A2->is_scalar());
      EXPECT_FALSE(A2->is_seqs_scalar());
      EXPECT_FALSE(A2->is_seqs());
      EXPECT_EQ(A2->shape(), Shape({1,1,1}));
    }

    ArrayPtr A3 = ones(MPI_COMM_WORLD, {1,1,1}, 1, DATA_INT);
    if(rank == 0){
      EXPECT_TRUE(A3->is_scalar());
      EXPECT_FALSE(A3->is_seqs_scalar());
      EXPECT_FALSE(A3->is_seqs());
      EXPECT_EQ(A3->shape(), Shape({1,1,1}));
    }

    ArrayPtr A4 = ones(MPI_COMM_SELF, {1,1,1}, 1, DATA_INT);
    if(rank == 0){
      EXPECT_TRUE(A4->is_scalar());
      EXPECT_TRUE(A4->is_seqs_scalar());
      EXPECT_TRUE(A4->is_seqs());
      EXPECT_EQ(A4->shape(), Shape({1,1,1}));
    }

  }


  //test save and load function
  TEST_P(MPITest, InputAndOutput){
    ArrayPtr A = oa::funcs::seqs(MPI_COMM_WORLD, {m,n,p}, 1);
    oa::io::save(A, "/tmp/A.nc", "data");

    // const boost::filesystem::path fileName("/tmp/A.nc");

    // bool b = boost::filesystem::exists(boost::filesystem::status(fileName));
    // std::cout<<b<<std::endl;
    
    if(rank == 0){
      std::ifstream infile("/tmp/A.nc");
      EXPECT_TRUE(infile.good());      
    }
      
    ArrayPtr B = oa::io::load("/tmp/A.nc", "data", MPI_COMM_WORLD);

    ArrayPtr A1 = to_rank0(A);
    ArrayPtr B1 = to_rank0(B);
    
    if(rank == 0){
      EXPECT_TRUE(oa::funcs::is_equal(A1, B1));
    }
  }
  
  TEST_P(MPITest, ArrayCreation){
    ///:for t in dtypes
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


  TEST_P(MPITest, BasicMath_Arrray_Array){
    ///:for t1 in dtypes
    ///:for t2 in dtypes
    {
      DataType dt1  = oa::utils::dtype<${t1}$>::type;
      DataType dt2  = oa::utils::dtype<${t2}$>::type;
      
      NodePtr N1 = oa::ops::new_node(oa::funcs::seqs(comm, {m, n, p}, 0, dt1));
      NodePtr N2 = oa::ops::new_node(oa::funcs::consts(comm,
                                                       {m, n, p},
                                                       ${t2}$(2.0), 0));

      
      typedef otype<${t1}$, ${t2}$>::value result_type;
      arma::Cube<result_type> C3;
      arma::Cube<result_type> C1 = make_seqs<result_type>(m, n, p);
      arma::Cube<result_type> C2(m,n,p);
      C2.fill(result_type(2));
      
      ///:for o in [['+','PLUS'], ['-', 'MINUS'], ['%','MULT'], ['/', 'DIVD']]
      {
        NodePtr N3 = oa::ops::new_node(TYPE_${o[1]}$, N1, N2);
        ArrayPtr A3 = oa::funcs::to_rank0(oa::ops::eval(N3));

        C3 = C1 ${o[0]}$ C2;

        // N1->get_data()->display("A1");
        // N2->get_data()->display("A2");

        if(rank == 0){
          // A3->display("A3");
          // std::cout<<"C3"<<std::endl<<C3<<std::endl;
          // std::cout<<"Operation:${o[1]}$"<<std::endl;
          
          EXPECT_TRUE(oa::funcs::is_equal(A3, C3));
        }
      }
      ///:endfor
    }
    ///:endfor
    ///:endfor
  }

  TEST_P(MPITest, BasicMath_Arrray_Scalar){
    ///:for t1 in dtypes
    ///:for t2 in dtypes
    {
      DataType dt1  = oa::utils::dtype<${t1}$>::type;
      DataType dt2  = oa::utils::dtype<${t2}$>::type;
      
      NodePtr N1 = oa::ops::new_node(oa::funcs::consts(comm, {m, n, p},
                                                       ${t1}$(3.0), 0));
      NodePtr N2 = oa::ops::new_node(oa::funcs::consts(MPI_COMM_SELF,
                                                       {1, 1, 1}, ${t2}$(2), 0));

      typedef otype<${t1}$, ${t2}$>::value result_type;
      arma::Cube<result_type> C3, C4;
      arma::Cube<result_type> C1 = arma::ones<arma::Cube<result_type> >(m, n, p) * 3.0;
      result_type  C2 = result_type(2);
      
      ///:for o in [['+','PLUS'], ['-', 'MINUS'], ['*','MULT'], ['/', 'DIVD']]
      {
      	NodePtr N3 = oa::ops::new_node(TYPE_${o[1]}$, N1, N2);
      	NodePtr N4 = oa::ops::new_node(TYPE_${o[1]}$, N2, N1);        
	ArrayPtr A3 = oa::funcs::to_rank0(oa::ops::eval(N3));
	ArrayPtr A4 = oa::funcs::to_rank0(oa::ops::eval(N4));
        
	C3 = C1 ${o[0]}$ C2;
        C4 = C2 ${o[0]}$ C1;
        
	if(rank == 0){
          // std::cout<<"${o[1]}$"<<"    ${o[0]}$"<<std::endl;
          // std::cout<<C3<<std::endl;
          // A4->display("A4");
          // MPI_Barrier(A4->get_partition()->get_comm());

	  EXPECT_TRUE(oa::funcs::is_equal(A3, C3));
	  EXPECT_TRUE(oa::funcs::is_equal(A4, C4));
	}
      }
      ///:endfor
    }
    ///:endfor
    ///:endfor
  }


  TEST_P(MPITest, GhostUpdate){
    ///:for t in dtypes
    {
      ArrayPtr A1 =
        oa::funcs::seqs(comm,{m*5, n*5, p*5}, 1, oa::utils::dtype<${t}$>::type);

      // ArrayPtr A2 =
      //   oa::funcs::seqs(comm,{m*5, n*5, p*5}, 2, oa::utils::dtype<${t}$>::type);

      std::vector<MPI_Request> reqs; 
      update_ghost_start(A1, reqs, -1);
      update_ghost_end(reqs);
      reqs.clear();
      
      // update_ghost_start(A2, reqs, -1);
      // update_ghost_end(reqs);
      // reqs.clear();

      // update_ghost_start(A3, reqs, -1);
      // update_ghost_end(reqs);
      // reqs.clear();
      
      // if(A1->local_size() > 0){
      //   arma::Cube<${t}$> C1 = oa::utils::make_cube<${t}$>(A1->buffer_shape(),
      //                                                      A1->get_buffer());
      // }
    }
    ///:endfor
  }

  INSTANTIATE_TEST_CASE_P(OpenArray, MPITest,
                          gt::Combine(gt::Values(MPI_COMM_WORLD),
                                      MRange, NRange, PRange));
  
}
