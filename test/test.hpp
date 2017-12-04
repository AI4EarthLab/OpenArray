#ifndef __TEST_HPP__
#define __TEST_HPP__

#include "../Range.hpp"
#include "../Box.hpp"
#include "../Partition.hpp"
#include "../Array.hpp"
#include "../Function.hpp"
#include "../Internal.hpp"
#include "../Operator.hpp"
#include "../IO.hpp"
#include "../c-interface/c_oa_type.hpp"

#include <assert.h>

void test_Range() {
  cout<<endl;
  cout<<"-----------------------test_Range----------------------------"<<endl;
  // A [0, 0)
  Range A;
  A.display("A");
  assert(A.equal(0, -1));
  
  // B [1, 3)
  Range B(1, 2);
  B.display("B");
  cout<<"Range B's size is "<<B.size()<<endl;
  assert(B.equal(1, 2));
  assert(B.size() == 2);
  assert(!A.equal(B));
    
    // A [1, 1)
  A.shift(1);
  cout<<"After shift by 1: "<<endl;
  A.display("A");
  assert(A.is_inside(B));
  
  // A [2, 4)
  Range C(2, 3);
  C.display("C");
  cout<<"C & B has intersection"<<endl;
  assert(C.intersection(B));

  // C [2, 3)
  Range D(2, 2);
  assert(C.get_intersection(B).equal(D));
  cout<<"------------------------------------------------------------"<<endl;
  cout<<endl;
}

void test_Box() {
  cout<<endl;
  cout<<"-----------------------test_Box----------------------------"<<endl;
  Box A;
  A.display("A");

  int starts[3] = {0, 0, 0};
  int counts[3] = {5, 5, 5};
  Box B(starts, counts);
  B.display("B");

  Box C(1, 5, 1, 5, 1, 5);
  C.display("C");
  Shape s = C.shape();
  printf("C's shape is [%d, %d, %d]\n", s[0], s[1], s[2]);
  printf("C's size is %d\n", C.size());
  assert(!B.equal(C));
  assert(B.equal_shape(C));
  assert(A.is_inside(B));

  Box D = B.get_intersection(C);
  Box E = C.get_intersection(B);
  assert(D.equal(E));
  assert(B.intersection(C));
  D.display("B and C");
  cout<<"-----------------------------------------------------------"<<endl;
  cout<<endl;
}

void test_Partition() {
  cout<<endl;
  cout<<"-----------------------test_Partition----------------------------"<<endl;
  Partition A;
  A.display("A");
  Partition B(MPI_COMM_WORLD, 4, {4, 4, 4});
  B.display("B");
  Partition C(MPI_COMM_WORLD, {2, 2}, {2, 2}, {4});
  C.display("C");
  assert(B.equal(C));
  Shape B_shape = B.shape();
  printf("B_shape is [%d %d %d]\n", B_shape[0], B_shape[1], B_shape[2]);
  assert(B.size() == 4 * 4 * 4);
  B.set_stencil(1, 1);
  assert(!B.equal(C));
  Box box = B.get_local_box({0, 0, 0});
  box.display("box at [0, 0, 0]");
  box = B.get_local_box(1);
  box.display("box at [1, 0, 0]");
  assert(B.get_procs_rank(0, 1, 0) == 2);
  cout<<"-----------------------------------------------------------------"<<endl;
  cout<<endl;
}

void test_Array() {
  cout<<endl;
  cout<<"-----------------------test_Array----------------------------"<<endl;
  Partition par(MPI_COMM_WORLD, 4, {4, 4, 4});
  PartitionPtr par_ptr = make_shared<Partition>(par);
  Array A(par_ptr);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) A.get_partition()->display("A");
  cout<<"-------------------------------------------------------------"<<endl;
  cout<<endl;
}

void test_Pool() {
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 3, 2}, 1);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ap->display("Array seqs");
  
  ap = oa::funcs::consts(MPI_COMM_WORLD, {2, 2}, {2, 2}, {4}, 1, 1);
  ap->display("Array_Consts_m2");

  ap = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 4}, 1, 2);
  ap->display("Array_ones");
  
  ap = oa::funcs::zeros(MPI_COMM_WORLD, {4, 4, 4}, 1, 1);
  ap->display("Array_zeros"); 

  ap = oa::funcs::rand(MPI_COMM_WORLD, {4, 4, 4}, 1, 1);
  ap->display("rand");
}

void test_sub() {
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 4}, 1);
  Box box(1, 2, 2, 3, 1, 3);
  ap->display("======A======");
  //PartitionPtr pp = ap->get_partition()->sub(box);
  //if (pp->rank() == 0) pp->display("Sub Partition");
  ArrayPtr sub_ap = oa::funcs::subarray(ap, box);
  sub_ap->display("====sub_A=====");
}

// need 6 mpi_process
void test_transfer() {
  // A
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6}, {6, 6, 6}, {1}, 1);
  ap->display("======A======");
  // sub1
  Box box1(4, 6, 5, 12, 0, 0);
  ArrayPtr sub_ap1 = oa::funcs::subarray(ap, box1);
  sub_ap1->display("====sub_1=====");
  // sub2
  Box box2(8, 10, 6, 13, 0, 0);
  ArrayPtr sub_ap2 = oa::funcs::subarray(ap, box2);
  sub_ap2->display("====sub_2=====");

  ArrayPtr tf_ap = oa::funcs::transfer(sub_ap1, sub_ap2->get_partition());
  tf_ap->display("======transfer_array======");
}

void test_update_ghost() {
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 4}, 1);
  oa::internal::set_ghost_consts((int*)ap->get_buffer(), ap->local_shape(), 0, 1);
  int rk = ap->rank();
  //int size = ap->get_local_box().size(1);
  //oa::internal::set_buffer_consts((int*)ap->get_buffer(), size, rk);

  ap->display("A");

  Shape sp = ap->local_shape();
  sp[0] += 2;
  sp[1] += 2;
  sp[2] += 2;

  ap->get_partition()->set_stencil_type(STENCIL_BOX);

  vector<MPI_Request> reqs;
  oa::funcs::update_ghost_start(ap, reqs, -1);
  oa::funcs::update_ghost_end(reqs);

  oa::utils::mpi_order_start(MPI_COMM_WORLD);
  printf("=====%d======\n", rk);
  oa::utils::print_data(ap->get_buffer(), sp, DATA_INT);
  oa::utils::mpi_order_end(MPI_COMM_WORLD);
}

void test_operator() {
  NodePtr np1 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 3);
  np1->display("===A===");
  
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  NodePtr np2 = oa::ops::new_node(ap);
  np2->display("===B===");
  /*std::cout<<"int size: "<<dtype<int>::size<<std::endl;
  std::cout<<"bool size: "<<dtype<bool>::size<<std::endl;
  std::cout<<"double size: "<<dtype<double>::size<<std::endl;

  std::cout<<"int type: "<<dtype<int>::type<<std::endl;
  std::cout<<"bool type: "<<dtype<bool>::type<<std::endl;
  std::cout<<"double type: "<<dtype<double>::type<<std::endl;
  */
  //ap->get_data()->display("======A======");
}


void test_io() {
  ArrayPtr A = oa::funcs::seqs(MPI_COMM_WORLD, {4,4,1}, 1);
  A->display("A:");
  oa::io::save(A, "A.nc", "data");

  ArrayPtr B = oa::io::load("A.nc", "data", MPI_COMM_WORLD);
  B->display("B:");
}

void test_write_graph() {
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
  // ((A+B)-(C*D))/E
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);
  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);
  NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr G = oa::ops::new_node(TYPE_MULT, C, D);
  NodePtr H = oa::ops::new_node(TYPE_MINUS, F, G);
  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, E);

  oa::ops::write_graph(I);

}

void test_force_eval() {
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3.0, 1);
  // ((A+B)-(C*D))/E
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);

  A->display("A");
  B->display("B");
  C->display("C");

  NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr G = oa::ops::new_node(TYPE_PLUS, F, C);
  ArrayPtr ans = oa::ops::force_eval(G);
  ans->display("A+B+C");
  
  Box box1(0,2,0,2,0,0);
  Box box2(0,2,1,3,0,0);
  Box box3(1,3,1,3,0,0);
  ap1 = oa::funcs::subarray(ap1, box1);
  ap2 = oa::funcs::subarray(ap2, box2);
  ap3 = oa::funcs::subarray(ap3, box3);

  NodePtr SA = oa::ops::new_node(ap1);
  NodePtr SB = oa::ops::new_node(ap2);
  NodePtr SC = oa::ops::new_node(ap3);

  cout<<endl;

  SA->display("SA");
  SB->display("SB");
  SC->display("SC");

  NodePtr SF = oa::ops::new_node(TYPE_PLUS, SA, SB);
  NodePtr SG = oa::ops::new_node(TYPE_PLUS, SF, SC);
  ArrayPtr sans = oa::ops::force_eval(SG);
  sans->display("SA+SB+SC");

  //cout<<0/0<<endl;
  //cout<<1/0<<endl;

  NodePtr seq_B = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
  NodePtr seq_C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);

  SF = oa::ops::new_node(TYPE_PLUS, SA, seq_B);
  SG = oa::ops::new_node(TYPE_MINUS, SF, seq_C);
  sans = oa::ops::force_eval(SG);
  cout<<"-----------------"<<endl;
  sans->display("SA+seq_B-seq_C");
}

void test_fusion_kernel() {
  /*ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
  // ((A+B)-(C*D))/E
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);
  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
  NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr G = oa::ops::new_node(TYPE_MULT, C, D);
  NodePtr H = oa::ops::new_node(TYPE_MINUS, F, G);
  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, E);

  oa::ops::gen_kernels(I);

  ArrayPtr ans = oa::ops::eval(I);
  A->display("A");
  B->display("B");
  C->display("C");
  D->display("D");
  E->display("E");
  ans->display("((A+B)-(C*D)) / E");*/


  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
  // (A+B)*C
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);
  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
  
  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
  oa::ops::force_eval(I)->display("I");

  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);

  oa::ops::force_eval(J)->display("J");
  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);

  //oa::ops::gen_kernels(K,true,MPI_COMM_WORLD);

  ArrayPtr ans = oa::ops::force_eval(K);
  ans->display("force_eval");
/*  ArrayPtr ans = oa::ops::eval(I);
  A->display("A");
  B->display("B");
  C->display("C");
  D->display("D");
  E->display("E");
  ans->display("((A+B)-(C*D)) / E");*/
}

void test_c_interface() {
/*  void* ap1 = seqs(4,4,1,1);
  void* ap2 = ones(4,4,1,1);
  void* ap3 = consts_int(4,4,1,3,1);
  
  void* A = new_node_array(ap1);
  void* B = new_node_array(ap2);
  void* C = new_node_array(ap3);
  void* D = new_seqs_scalar_node_int(1, MPI_COMM_SELF);
  void* E = new_seqs_scalar_node_double(2.0, MPI_COMM_SELF);
  void* F = new_node_op2(TYPE_PLUS, A, B);
  void* G = new_node_op2(TYPE_MULT, C, D);
  void* H = new_node_op2(TYPE_MINUS, F, G);
  void* I = new_node_op2(TYPE_DIVD, H, E);

  ArrayPtr ans = oa::ops::eval(*(NodePtr*)I);
  ans->display();*/
  void* A = NULL;
  ones(A, 4, 4, 1, 1);
  (*(ArrayPtr*)A)->display();
  destroy_array(A);
  cout<<"after "<<A<<endl;
}

void test_logic_operator() {
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3.0, 1);
  
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(TYPE_GT, A, B);

  ArrayPtr ans = oa::ops::eval(C);
  ap1->display("A");
  ap2->display("B");
  ans->display("A>B");

  C = oa::ops::new_node(TYPE_GE, A, B);
  ans = oa::ops::eval(C);
  ans->display("A>=B");

  C = oa::ops::new_node(TYPE_LT, A, B);
  ans = oa::ops::eval(C);
  ans->display("A<B");

  C = oa::ops::new_node(TYPE_LE, A, B);
  ans = oa::ops::eval(C);
  ans->display("A<=B");

  C = oa::ops::new_node(TYPE_EQ, A, B);
  ans = oa::ops::eval(C);
  ans->display("A==B");

  C = oa::ops::new_node(TYPE_NE, A, B);
  ans = oa::ops::eval(C);
  ans->display("A!=B");

  C = oa::ops::new_node(TYPE_OR, A, B);
  ans = oa::ops::eval(C);
  ans->display("A||B");

  C = oa::ops::new_node(TYPE_AND, A, B);
  ans = oa::ops::eval(C);
  ans->display("A&&B");

  C = oa::ops::new_node(TYPE_NOT, A);
  ans = oa::ops::eval(C);
  ans->display("!A");
}

void test_math_operator() {
  ArrayPtr ap1 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 0.5, 1);
  
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(TYPE_EXP, A);

  ArrayPtr ans = oa::ops::eval(B);
  ap1->display("A");
  ans->display("exp(A)");

  B = oa::ops::new_node(TYPE_SIN, A);
  ans = oa::ops::eval(B);
  ans->display("sin(A)");

  B = oa::ops::new_node(TYPE_COS, A);
  ans = oa::ops::eval(B);
  ans->display("cos(A)");

  B = oa::ops::new_node(TYPE_TAN, A);
  ans = oa::ops::eval(B);
  ans->display("tan(A)");

  B = oa::ops::new_node(TYPE_RCP, A);
  ans = oa::ops::eval(B);
  ans->display("rcp(A)");

  B = oa::ops::new_node(TYPE_SQRT, A);
  ans = oa::ops::eval(B);
  ans->display("sqrt(A)");

  B = oa::ops::new_node(TYPE_ASIN, A);
  ans = oa::ops::eval(B);
  ans->display("asin(A)");

  B = oa::ops::new_node(TYPE_ACOS, A);
  ans = oa::ops::eval(B);
  ans->display("acos(A)");

  B = oa::ops::new_node(TYPE_ATAN, A);
  ans = oa::ops::eval(B);
  ans->display("atan(A)");

  B = oa::ops::new_node(TYPE_ABS, A);
  ans = oa::ops::eval(B);
  ans->display("abs(A)");

  B = oa::ops::new_node(TYPE_LOG, A);
  ans = oa::ops::eval(B);
  ans->display("log(A)");

  B = oa::ops::new_node(TYPE_LOG10, A);
  ans = oa::ops::eval(B);
  ans->display("log10(A)");

  B = oa::ops::new_node(TYPE_TANH, A);
  ans = oa::ops::eval(B);
  ans->display("tanh(A)");

  B = oa::ops::new_node(TYPE_SINH, A);
  ans = oa::ops::eval(B);
  ans->display("sinh(A)");

  B = oa::ops::new_node(TYPE_COSH, A);
  ans = oa::ops::eval(B);
  ans->display("cosh(A)");

  B = oa::ops::new_node(TYPE_UPLUS, A);
  ans = oa::ops::eval(B);
  ans->display("+(A)");

  B = oa::ops::new_node(TYPE_UMINUS, A);
  ans = oa::ops::eval(B);
  ans->display("-(A)");

  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
  B = oa::ops::new_node(TYPE_POW, A, C);
  ans = oa::ops::eval(B);
  ans->display("pow(B,A)");

}

void test_gen_kernel_JIT() {
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
  // (A+B)*C
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);
  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
  
  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);
  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);

  oa::ops::gen_kernels_JIT(K, true, MPI_COMM_WORLD);

  G = oa::ops::new_node(TYPE_PLUS, A, B);
  H = oa::ops::new_node(TYPE_MULT, G, C);
  I = oa::ops::new_node(TYPE_DIVD, H, D);
  J = oa::ops::new_node(TYPE_MINUS, E, F);
  K = oa::ops::new_node(TYPE_PLUS, I, J);
  
  oa::ops::gen_kernels_JIT(K, true, MPI_COMM_WORLD);
}

void test_min_max() {
  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 7, 3}, 3);
  NodePtr EA2 = oa::ops::new_node(eap2);
  EA2->display("EA2");
  NodePtr EC2 = oa::ops::new_node(TYPE_MAX, EA2);

  ArrayPtr esans2 = oa::ops::eval(EC2);
  esans2->display("max(EA2)");
}

void test_csum() {
  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 8, 8}, 1);
  NodePtr EA2 = oa::ops::new_node(eap2);
  EA2->display("EA2");
  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
  NodePtr EC2 = oa::ops::new_node(TYPE_CSUM, EA2, C);

  ArrayPtr esans2 = oa::ops::eval(EC2);
  esans2->display("csum(EA2)");
}
void test_sum() {
  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 8, 8}, 1);
  NodePtr EA2 = oa::ops::new_node(eap2);
  EA2->display("EA2");
  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
  NodePtr EC2 = oa::ops::new_node(TYPE_SUM, EA2, C);

  ArrayPtr esans2 = oa::ops::eval(EC2);
  esans2->display("sum(EA2)");
}

void test_eval() {
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
  // (A+B)*C
  NodePtr A = oa::ops::new_node(ap1);
  NodePtr B = oa::ops::new_node(ap2);
  NodePtr C = oa::ops::new_node(ap3);
  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
  
  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);
  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);

  oa::ops::gen_kernels_JIT(K, true, MPI_COMM_WORLD);

  A->display("A");
  B->display("B");
  C->display("C");
  D->display("D");
  E->display("E");
  F->display("F");

  ArrayPtr ans = oa::ops::eval(K);
  ans->display("eval");
}

// need 6 mpi_process
void test_set() {
  // A
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6}, {6, 6, 6}, {1}, 1);
  ap->display("======A======");
  // sub1
  Box box1(4, 6, 5, 12, 0, 0);
  
  // sub2
  Box box2(8, 10, 6, 13, 0, 0);
  ArrayPtr sub_ap2 = oa::funcs::subarray(ap, box2);
  sub_ap2->display("====sub_2=====");

  //oa::funcs::set(ap, box1, sub_ap2);
  //ap->display("======after_set======");

  oa::funcs::set(ap, box1, ap, box2);
  ap->display("======after_set======");

  oa::funcs::set(ap, box1, 0);
  ap->display("======after_set======");
}
/*
void test_set_l2g() {
  // A
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_SELF, {4, 4, 4}, 1);
  ap->display("======A======");
 
}
*/

void test_rep() {
  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 4}, 1);
  ap->display("======A======");

  ArrayPtr repA = oa::funcs::rep(ap, 2, 2, 2);
  repA->display("======after_rep======");
  ap->display("======A======");
}

#endif
