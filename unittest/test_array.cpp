#include <armadillo>
#include "../Array.hpp"
#include "../Function.hpp"
#include "../NodePool.hpp"
#include "../Operator.hpp"
#include "../IO.hpp"
#include "gtest/gtest.h"
#include <boost/filesystem.hpp>
#include <unistd.h>

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
///:set fdtypes = ['float', 'double']

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


  TEST_P(MPITest, MinMax){

    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;
      ArrayPtr A = oa::funcs::rands(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);
      //NA->display("NA");

      NodePtr N1 = oa::ops::new_node(TYPE_MAX, NA);
      NodePtr N2 = oa::ops::new_node(TYPE_MIN, NA);
      NodePtr N3 = oa::ops::new_node(TYPE_ABS_MAX, NA);
      NodePtr N4 = oa::ops::new_node(TYPE_ABS_MIN, NA);
      
      NodePtr N5 = oa::ops::new_node(TYPE_MAX_AT, NA);
      NodePtr N6 = oa::ops::new_node(TYPE_MIN_AT, NA);
      NodePtr N7 = oa::ops::new_node(TYPE_ABS_MAX_AT, NA);
      NodePtr N8 = oa::ops::new_node(TYPE_ABS_MIN_AT, NA);

      ArrayPtr V1 = oa::funcs::to_rank0(oa::ops::eval(N1));
      ArrayPtr V2 = oa::funcs::to_rank0(oa::ops::eval(N2));
      ArrayPtr V3 = oa::funcs::to_rank0(oa::ops::eval(N3));
      ArrayPtr V4 = oa::funcs::to_rank0(oa::ops::eval(N4));
      ArrayPtr V5 = oa::funcs::to_rank0(oa::ops::eval(N5));
      ArrayPtr V6 = oa::funcs::to_rank0(oa::ops::eval(N6));
      ArrayPtr V7 = oa::funcs::to_rank0(oa::ops::eval(N7));
      ArrayPtr V8 = oa::funcs::to_rank0(oa::ops::eval(N8));
      
      // NodePtr NSA = oa::ops::new_node(TYPE_MIN, NA);

      //A->display("A");
      
      ArrayPtr A1 = oa::funcs::to_rank0(A);
      Shape s = A1->buffer_shape();
      arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(
          s, A1->get_buffer())(
              arma::span(sw, s[0] - sw - 1),
              arma::span(sw, s[1] - sw - 1),
              arma::span(sw, s[2] - sw - 1));
      

      // ArrayPtr V2 = oa::ops::eval(NSA);
      
      if(rank == 0){

        EXPECT_TRUE(oa::funcs::is_equal(V1, C.max()));
        EXPECT_TRUE(oa::funcs::is_equal(V2, C.min()));
        EXPECT_TRUE(oa::funcs::is_equal(V3, arma::abs(C).max()));
        EXPECT_TRUE(oa::funcs::is_equal(V4, arma::abs(C).min()));

        
        // V1->display("V1");
        // std::cout<<"C.max():"<<C.max()<<std::endl;
        // V2->display("V2");
        // std::cout<<"C.min():"<<C.min()<<std::endl;
        // V1->display("V3");
        // std::cout<<"abs(C).max():"<<arma::abs(C).max()<<std::endl;
        // V2->display("V4");
        // std::cout<<"abs(C).min():"<<arma::abs(C).min()<<std::endl;

        arma::uvec VI;
        VI = ind2sub(arma::size(C), C.index_max());

        EXPECT_TRUE(oa::funcs::is_equal(V5, VI.memptr()));

        VI = ind2sub(arma::size(C), C.index_min());
        
        // V6->display("V6");
        // std::cout<<VI<<std::endl;

        EXPECT_TRUE(oa::funcs::is_equal(V6, VI.memptr()));

        VI = ind2sub(arma::size(C), abs(C).index_max());
        EXPECT_TRUE(oa::funcs::is_equal(V7, VI.memptr()));

        VI = ind2sub(arma::size(C), abs(C).index_min());
        EXPECT_TRUE(oa::funcs::is_equal(V8, VI.memptr()));
        
        // V2->display("V2");
        //EXPECT_TRUE(V1->is_seqs());
        //EXPECT_TRUE(V1->shape() == SCALAR_SHAPE);
      }
    }
    ///:endfor
  }

  TEST_P(MPITest, SUM_scalar_CSUM_scalar){

    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 0);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_CSUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);
      ${t}$* res = (${t}$*) RA0->get_buffer();

      NodePtr N1 = oa::ops::new_node(TYPE_SUM, NA, type0);
      ArrayPtr RA1 = oa::ops::eval(N1);
      ${t}$* res1 = (${t}$*) RA1->get_buffer();



      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        x = accu(C);
        EXPECT_TRUE(res[0] == x);
        EXPECT_TRUE(res1[0] == x);
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, CSUM_x){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_CSUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = 0; i <= m; i++){
          if(i-1 >= 0 && i <= m-1){
            C.subcube( i, 0, 0, i, n-1, p-1 ) += C.subcube( i-1, 0, 0, i-1, n-1, p-1 );
          }
        }
        EXPECT_TRUE(oa::funcs::is_equal(result, C));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, CSUM_y){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_CSUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = 0; i <= n ; i++){
          if(i-1 >= 0 && i <= n-1){
            C.subcube( 0, i, 0, m-1, i, p-1 ) += C.subcube( 0, i-1, 0, m-1, i-1, p-1 );
          }
        }
        EXPECT_TRUE(oa::funcs::is_equal(result, C));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, CSUM_z){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 3);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_CSUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = 0; i <=p; i++){
          if(i-1 >= 0 && i <= p-1){
            C.subcube( 0, 0, i, m-1, n-1, i ) += C.subcube( 0, 0, i-1, m-1, n-1, i-1 );
          }
        }
        EXPECT_TRUE(oa::funcs::is_equal(result, C));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, SUM_x){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_SUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = m; i >= 0; i--){
          if(i-1 >= 0 && i <= m-1){
            C.subcube( 0, 0, 0, 0, n-1, p-1 ) += C.subcube( i, 0, 0, i, n-1, p-1 );
          }
        }
        arma::Cube<${t}$> Cr = C.subcube( 0, 0, 0, 0, n-1, p-1 );
        EXPECT_TRUE(oa::funcs::is_equal(result, Cr));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, SUM_y){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_SUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = n; i >= 0; i--){
          if(i-1 >= 0 && i <= n-1){
            C.subcube( 0, 0, 0, m-1, 0, p-1 ) += C.subcube( 0, i, 0, m-1, i, p-1 );
          }
        }
        arma::Cube<${t}$> Cr = C.subcube( 0, 0, 0, m-1, 0, p-1 );
        EXPECT_TRUE(oa::funcs::is_equal(result, Cr));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, SUM_z){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      double x = 0;
      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NA = oa::ops::new_node(A);

      NodePtr type0 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 3);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
      NodePtr N0 = oa::ops::new_node(TYPE_SUM, NA, type0);
      ArrayPtr RA0 = oa::ops::eval(N0);


      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(RA0);
      if(rank == 0){
        arma::Cube<${t}$> C = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        for(int i = p; i >= 0; i--){
          if(i-1 >= 0 && i <= p-1){
            C.subcube( 0, 0, 0, m-1, n-1, 0 ) += C.subcube( 0, 0, i, m-1, n-1, i );
          }
        }
        arma::Cube<${t}$> Cr = C.subcube( 0, 0, 0, m-1, n-1, 0 );
        EXPECT_TRUE(oa::funcs::is_equal(result, Cr));
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, REP){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      int x = 2;
      int y = 2;
      int z = 2;

      ArrayPtr A = oa::funcs::seqs(comm, {m,n,p}, sw, dt);
      NodePtr NN = oa::ops::new_node(A);
      ArrayPtr lap = oa::funcs::consts(MPI_COMM_SELF, {3, 1, 1}, 2, 0);
      NodePtr NN2 = oa::ops::new_node(lap);


      NodePtr NP = oa::ops::new_node(TYPE_REP, NN, NN2);
      ArrayPtr repA = oa::ops::eval(NP);

      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      ArrayPtr result = oa::funcs::to_rank0(repA);
      //if(rank == 0)result->display("result");
      if(rank == 0){
        arma::Cube<${t}$> C0 = oa::utils::make_cube<${t}$>(rank0A->buffer_shape(), rank0A->get_buffer());
        arma::Cube<${t}$> Cr(m*x, n*y, p*z); 
        Cr.zeros();
        int ii,jj,kk;
        //result->display("result");
        //Cr.print("Cr");
        ii = 0;
        for(int i = 0; i < x; i++){
          jj = 0;
          for(int j = 0; j < y; j++){
            kk = 0;
            for(int k = 0; k < z; k++){
              //cout<<ii<<","<<jj<<","<<kk<<endl;
              Cr.subcube(0+ii,0+jj,0+kk,m-1+ii,n-1+jj,p-1+kk) = C0;
              kk += p;
            }
            jj += n;
          }
          ii += m;
        }
        //Cr.print("Cr");
        //result->display("result");
        //        EXPECT_TRUE(oa::funcs::is_equal(rank0A, C));
        EXPECT_TRUE(oa::funcs::is_equal(result, Cr));

      }


      MPI_Barrier(comm);
    }
    ///:endfor
  }

  TEST_P(MPITest, RAND){
    ///:for t in dtypes
    {
      int sw = NO_STENCIL;
      DataType dt = oa::utils::dtype<${t}$>::type;

      int x = 2;
      int y = 3;
      int z = 4;

      ArrayPtr A = oa::funcs::rands(comm, {m,n,p}, sw, dt);

      ArrayPtr rank0A = oa::funcs::to_rank0(A);
      //if(rank == 0)result->display("result");
      if(rank == 0){
        //rank0A->display("rand");
        ;
      }

      MPI_Barrier(comm);
    }
    ///:endfor
  }


  INSTANTIATE_TEST_CASE_P(OpenArray, MPITest,
          gt::Combine(gt::Values(MPI_COMM_WORLD),
                  MRange, NRange, PRange));

}
