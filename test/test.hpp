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
#include "../op_define.hpp"
#include "../Grid.hpp"
#include "../modules/basic/new_node.hpp"
#include "../modules/operator/new_node.hpp"
#include "../utils/utils.hpp"
#include "../cache.hpp"
#include "../MPI.hpp"
#include <assert.h>

//void test_Range() {
//  cout<<endl;
//  cout<<"-----------------------test_Range----------------------------"<<endl;
//  // A [0, 0)
//  Range A;
//  A.display("A");
//  assert(A.equal(0, -1));
//  
//  // B [1, 3)
//  Range B(1, 2);
//  B.display("B");
//  cout<<"Range B's size is "<<B.size()<<endl;
//  assert(B.equal(1, 2));
//  assert(B.size() == 2);
//  assert(!A.equal(B));
//    
//  // A [1, 1)
//  A.shift(1);
//  cout<<"After shift by 1: "<<endl;
//  A.display("A");
//  assert(A.is_inside(B));
//  
//  // A [2, 4)
//  Range C(2, 3);
//  C.display("C");
//  cout<<"C & B has intersection"<<endl;
//  assert(C.intersection(B));
//
//  // C [2, 3)
//  Range D(2, 2);
//  assert(C.get_intersection(B).equal(D));
//  cout<<"------------------------------------------------------------"<<endl;
//  cout<<endl;
//}
//
//void test_Box() {
//  cout<<endl;
//  cout<<"-----------------------test_Box----------------------------"<<endl;
//  Box A;
//  A.display("A");
//
//  int starts[3] = {0, 0, 0};
//  int counts[3] = {5, 5, 5};
//  Box B(starts, counts);
//  B.display("B");
//
//  Box C(1, 5, 1, 5, 1, 5);
//  C.display("C");
//  Shape s = C.shape();
//  printf("C's shape is [%d, %d, %d]\n", s[0], s[1], s[2]);
//  printf("C's size is %d\n", C.size());
//  assert(!B.equal(C));
//  assert(B.equal_shape(C));
//  assert(A.is_inside(B));
//
//  Box D = B.get_intersection(C);
//  Box E = C.get_intersection(B);
//  assert(D.equal(E));
//  assert(B.intersection(C));
//  D.display("B and C");
//  cout<<"-----------------------------------------------------------"<<endl;
//  cout<<endl;
//}
//
//void test_Partition() {
//  cout<<endl;
//  cout<<"-----------------------test_Partition----------------------------"<<endl;
//  Partition A;
//  A.display("A");
//  Partition B(MPI_COMM_WORLD, 4, {4, 4, 4});
//  B.display("B");
//  Partition C(MPI_COMM_WORLD, {2, 2}, {2, 2}, {4});
//  C.display("C");
//  assert(B.equal(C));
//  Shape B_shape = B.shape();
//  printf("B_shape is [%d %d %d]\n", B_shape[0], B_shape[1], B_shape[2]);
//  assert(B.size() == 4 * 4 * 4);
//  B.set_stencil(1, 1);
//  assert(!B.equal(C));
//  Box box = B.get_local_box({0, 0, 0});
//  box.display("box at [0, 0, 0]");
//  box = B.get_local_box(1);
//  box.display("box at [1, 0, 0]");
//  assert(B.get_procs_rank(0, 1, 0) == 2);
//  cout<<"-----------------------------------------------------------------"<<endl;
//  cout<<endl;
//}
//
//void test_Array() {
//  cout<<endl;
//  cout<<"-----------------------test_Array----------------------------"<<endl;
//  Partition par(MPI_COMM_WORLD, 4, {4, 4, 4});
//  PartitionPtr par_ptr = make_shared<Partition>(par);
//  Array A(par_ptr);
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  if (rank == 0) A.get_partition()->display("A");
//  cout<<"-------------------------------------------------------------"<<endl;
//  cout<<endl;
//}
//
//void test_Pool() {
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 3, 2}, 1);
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  ap->display("Array seqs");
//  
//  ap = oa::funcs::consts(MPI_COMM_WORLD, {2, 2}, {2, 2}, {4}, 1, 1);
//  ap->display("Array_Consts_m2");
//
//  ap = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 4}, 1, 2);
//  ap->display("Array_ones");
//  
//  ap = oa::funcs::zeros(MPI_COMM_WORLD, {4, 4, 4}, 1, 1);
//  ap->display("Array_zeros"); 
//
//  ap = oa::funcs::rands(MPI_COMM_WORLD, {4, 4, 4}, 1, 1);
//  ap->display("rand");
//}
//
//void test_sub() {
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 4}, 1);
//  Box box(1, 2, 2, 3, 1, 3);
//  ap->display("======A======");
//  //PartitionPtr pp = ap->get_partition()->sub(box);
//  //if (pp->rank() == 0) pp->display("Sub Partition");
//  ArrayPtr sub_ap = oa::funcs::subarray(ap, box);
//  sub_ap->display("====sub_A=====");
//}
//
//// need 6 mpi_process
//void test_transfer() {
//  // A
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6}, {6, 6, 6}, {1}, 1);
//  ap->display("======A======");
//  // sub1
//  Box box1(4, 6, 5, 12, 0, 0);
//  ArrayPtr sub_ap1 = oa::funcs::subarray(ap, box1);
//  sub_ap1->display("====sub_1=====");
//  // sub2
//  Box box2(8, 10, 6, 13, 0, 0);
//  ArrayPtr sub_ap2 = oa::funcs::subarray(ap, box2);
//  sub_ap2->display("====sub_2=====");
//
//  ArrayPtr tf_ap = oa::funcs::transfer(sub_ap1, sub_ap2->get_partition());
//  tf_ap->display("======transfer_array======");
//}
//
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
  oa::funcs::update_ghost_start(ap, reqs, 4, {1,0,0}, {0,1,0});
  oa::funcs::update_ghost_end(reqs);

  MPI_ORDER_START
  printf("=====%d======\n", rk);
  oa::utils::print_data(ap->get_buffer(), sp, DATA_INT);
  MPI_ORDER_END
}
//
//void test_operator() {
//  NodePtr np1 = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 3);
//  np1->display("===A===");
//  
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  NodePtr np2 = oa::ops::new_node(ap);
//  np2->display("===B===");
//  /*std::cout<<"int size: "<<dtype<int>::size<<std::endl;
//    std::cout<<"bool size: "<<dtype<bool>::size<<std::endl;
//    std::cout<<"double size: "<<dtype<double>::size<<std::endl;
//
//    std::cout<<"int type: "<<dtype<int>::type<<std::endl;
//    std::cout<<"bool type: "<<dtype<bool>::type<<std::endl;
//    std::cout<<"double type: "<<dtype<double>::type<<std::endl;
//  */
//  //ap->get_data()->display("======A======");
//}
//
//
//void test_io() {
//  ArrayPtr A = oa::funcs::seqs(MPI_COMM_WORLD, {4,4,1}, 1);
//  A->display("A:");
//  oa::io::save(A, "A.nc", "data");
//
//  ArrayPtr B = oa::io::load("A.nc", "data", MPI_COMM_WORLD);
//  B->display("B:");
//}
//
//void test_write_graph() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
//  // ((A+B)-(C*D))/E
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(ap3);
//  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
//  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);
//  NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
//  NodePtr G = oa::ops::new_node(TYPE_MULT, C, D);
//  NodePtr H = oa::ops::new_node(TYPE_MINUS, F, G);
//  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, E);
//
//  oa::ops::write_graph(I);
//
//}
//
//void test_force_eval() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3.0, 1);
//  // ((A+B)-(C*D))/E
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(ap3);
//
//  A->display("A");
//  B->display("B");
//  C->display("C");
//
//  NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
//  NodePtr G = oa::ops::new_node(TYPE_PLUS, F, C);
//  ArrayPtr ans = oa::ops::force_eval(G);
//  ans->display("A+B+C");
//  
//  Box box1(0,2,0,2,0,0);
//  Box box2(0,2,1,3,0,0);
//  Box box3(1,3,1,3,0,0);
//  ap1 = oa::funcs::subarray(ap1, box1);
//  ap2 = oa::funcs::subarray(ap2, box2);
//  ap3 = oa::funcs::subarray(ap3, box3);
//
//  NodePtr SA = oa::ops::new_node(ap1);
//  NodePtr SB = oa::ops::new_node(ap2);
//  NodePtr SC = oa::ops::new_node(ap3);
//
//  cout<<endl;
//
//  SA->display("SA");
//  SB->display("SB");
//  SC->display("SC");
//
//  NodePtr SF = oa::ops::new_node(TYPE_PLUS, SA, SB);
//  NodePtr SG = oa::ops::new_node(TYPE_PLUS, SF, SC);
//  ArrayPtr sans = oa::ops::force_eval(SG);
//  sans->display("SA+SB+SC");
//
//  //cout<<0/0<<endl;
//  //cout<<1/0<<endl;
//
//  NodePtr seq_B = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
//  NodePtr seq_C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2);
//
//  SF = oa::ops::new_node(TYPE_PLUS, SA, seq_B);
//  SG = oa::ops::new_node(TYPE_MINUS, SF, seq_C);
//  sans = oa::ops::force_eval(SG);
//  cout<<"-----------------"<<endl;
//  sans->display("SA+seq_B-seq_C");
//}
//
//void test_fusion_kernel() {
//  /*ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//    ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//    ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
//    // ((A+B)-(C*D))/E
//    NodePtr A = oa::ops::new_node(ap1);
//    NodePtr B = oa::ops::new_node(ap2);
//    NodePtr C = oa::ops::new_node(ap3);
//    NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);
//    NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
//    NodePtr F = oa::ops::new_node(TYPE_PLUS, A, B);
//    NodePtr G = oa::ops::new_node(TYPE_MULT, C, D);
//    NodePtr H = oa::ops::new_node(TYPE_MINUS, F, G);
//    NodePtr I = oa::ops::new_node(TYPE_DIVD, H, E);
//
//    oa::ops::gen_kernels(I);
//
//    ArrayPtr ans = oa::ops::eval(I);
//    A->display("A");
//    B->display("B");
//    C->display("C");
//    D->display("D");
//    E->display("E");
//    ans->display("((A+B)-(C*D)) / E");*/
//
//
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
//  // (A+B)*C
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(ap3);
//  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
//  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
//  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
//  
//  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
//  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
//  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
//  oa::ops::force_eval(I)->display("I");
//
//  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);
//
//  oa::ops::force_eval(J)->display("J");
//  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);
//
//  //oa::ops::gen_kernels(K,true);
//
//  ArrayPtr ans = oa::ops::force_eval(K);
//  ans->display("force_eval");
//  /*  ArrayPtr ans = oa::ops::eval(I);
//      A->display("A");
//      B->display("B");
//      C->display("C");
//      D->display("D");
//      E->display("E");
//      ans->display("((A+B)-(C*D)) / E");*/
//}
//
//void test_c_interface() {
//  /*  void* ap1 = seqs(4,4,1,1);
//      void* ap2 = ones(4,4,1,1);
//      void* ap3 = consts_int(4,4,1,3,1);
//  
//      void* A = new_node_array(ap1);
//      void* B = new_node_array(ap2);
//      void* C = new_node_array(ap3);
//      void* D = new_seqs_scalar_node_int(1, MPI_COMM_SELF);
//      void* E = new_seqs_scalar_node_double(2.0, MPI_COMM_SELF);
//      void* F = new_node_op2(TYPE_PLUS, A, B);
//      void* G = new_node_op2(TYPE_MULT, C, D);
//      void* H = new_node_op2(TYPE_MINUS, F, G);
//      void* I = new_node_op2(TYPE_DIVD, H, E);
//
//      ArrayPtr ans = oa::ops::eval(*(NodePtr*)I);
//      ans->display();*/
//}
//
//void test_logic_operator() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3.0, 1);
//  
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(TYPE_GT, A, B);
//
//  ArrayPtr ans = oa::ops::eval(C);
//  ap1->display("A");
//  ap2->display("B");
//  ans->display("A>B");
//
//  C = oa::ops::new_node(TYPE_GE, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A>=B");
//
//  C = oa::ops::new_node(TYPE_LT, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A<B");
//
//  C = oa::ops::new_node(TYPE_LE, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A<=B");
//
//  C = oa::ops::new_node(TYPE_EQ, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A==B");
//
//  C = oa::ops::new_node(TYPE_NE, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A!=B");
//
//  C = oa::ops::new_node(TYPE_OR, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A||B");
//
//  C = oa::ops::new_node(TYPE_AND, A, B);
//  ans = oa::ops::eval(C);
//  ans->display("A&&B");
//
//  C = oa::ops::new_node(TYPE_NOT, A);
//  ans = oa::ops::eval(C);
//  ans->display("!A");
//}
//
//void test_math_operator() {
//  ArrayPtr ap1 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 0.5, 1);
//  
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(TYPE_EXP, A);
//
//  ArrayPtr ans = oa::ops::eval(B);
//  ap1->display("A");
//  ans->display("exp(A)");
//
//  B = oa::ops::new_node(TYPE_SIN, A);
//  ans = oa::ops::eval(B);
//  ans->display("sin(A)");
//
//  B = oa::ops::new_node(TYPE_COS, A);
//  ans = oa::ops::eval(B);
//  ans->display("cos(A)");
//
//  B = oa::ops::new_node(TYPE_TAN, A);
//  ans = oa::ops::eval(B);
//  ans->display("tan(A)");
//
//  B = oa::ops::new_node(TYPE_RCP, A);
//  ans = oa::ops::eval(B);
//  ans->display("rcp(A)");
//
//  B = oa::ops::new_node(TYPE_SQRT, A);
//  ans = oa::ops::eval(B);
//  ans->display("sqrt(A)");
//
//  B = oa::ops::new_node(TYPE_ASIN, A);
//  ans = oa::ops::eval(B);
//  ans->display("asin(A)");
//
//  B = oa::ops::new_node(TYPE_ACOS, A);
//  ans = oa::ops::eval(B);
//  ans->display("acos(A)");
//
//  B = oa::ops::new_node(TYPE_ATAN, A);
//  ans = oa::ops::eval(B);
//  ans->display("atan(A)");
//
//  B = oa::ops::new_node(TYPE_ABS, A);
//  ans = oa::ops::eval(B);
//  ans->display("abs(A)");
//
//  B = oa::ops::new_node(TYPE_LOG, A);
//  ans = oa::ops::eval(B);
//  ans->display("log(A)");
//
//  B = oa::ops::new_node(TYPE_LOG10, A);
//  ans = oa::ops::eval(B);
//  ans->display("log10(A)");
//
//  B = oa::ops::new_node(TYPE_TANH, A);
//  ans = oa::ops::eval(B);
//  ans->display("tanh(A)");
//
//  B = oa::ops::new_node(TYPE_SINH, A);
//  ans = oa::ops::eval(B);
//  ans->display("sinh(A)");
//
//  B = oa::ops::new_node(TYPE_COSH, A);
//  ans = oa::ops::eval(B);
//  ans->display("cosh(A)");
//
//  B = oa::ops::new_node(TYPE_UPLUS, A);
//  ans = oa::ops::eval(B);
//  ans->display("+(A)");
//
//  B = oa::ops::new_node(TYPE_UMINUS, A);
//  ans = oa::ops::eval(B);
//  ans->display("-(A)");
//
//  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
//  B = oa::ops::new_node(TYPE_POW, A, C);
//  ans = oa::ops::eval(B);
//  ans->display("pow(B,A)");
//
//}
//
//void test_gen_kernel_JIT() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
//  // (A+B)*C
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(ap3);
//  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
//  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
//  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
//  
//  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
//  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
//  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
//  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);
//  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);
//
//  oa::ops::gen_kernels_JIT(K, true);
//
//  G = oa::ops::new_node(TYPE_PLUS, A, B);
//  H = oa::ops::new_node(TYPE_MULT, G, C);
//  I = oa::ops::new_node(TYPE_DIVD, H, D);
//  J = oa::ops::new_node(TYPE_MINUS, E, F);
//  K = oa::ops::new_node(TYPE_PLUS, I, J);
//  
//  oa::ops::gen_kernels_JIT(K, true);
//}
//
//void test_min_max() {
//  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 7, 3}, 3);
//  NodePtr EA2 = oa::ops::new_node(eap2);
//  EA2->display("EA2");
//  NodePtr EC2 = oa::ops::new_node(TYPE_MAX, EA2);
//
//  ArrayPtr esans2 = oa::ops::eval(EC2);
//  esans2->display("max(EA2)");
//}
//
//void test_csum() {
//  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 8, 8}, 1);
//  NodePtr EA2 = oa::ops::new_node(eap2);
//  EA2->display("EA2");
//  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 3);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
//  NodePtr EC2 = oa::ops::new_node(TYPE_CSUM, EA2, C);
//
//  ArrayPtr esans2 = oa::ops::eval(EC2);
//  esans2->display("csum(EA2)");
//}
//void test_sum() {
//  ArrayPtr eap2 = oa::funcs::seqs(MPI_COMM_WORLD, {8, 8, 8}, 1);
//  NodePtr EA2 = oa::ops::new_node(eap2);
//  EA2->display("EA2");
//  NodePtr C = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1);//c=0 scalar, c=1 sum to x, c=2 sum to y, c=3 sum to z
//  NodePtr EC2 = oa::ops::new_node(TYPE_SUM, EA2, C);
//
//  ArrayPtr esans2 = oa::ops::eval(EC2);
//  esans2->display("sum(EA2)");
//}
//
//void test_eval() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap2 = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 1}, 1);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {4, 4, 1}, 3, 1);
//  // (A+B)*C
//  NodePtr A = oa::ops::new_node(ap1);
//  NodePtr B = oa::ops::new_node(ap2);
//  NodePtr C = oa::ops::new_node(ap3);
//  NodePtr D = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 1.0);
//  NodePtr E = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, 2.0);
//  NodePtr F = oa::ops::new_seqs_scalar_node(MPI_COMM_SELF, float(1.0));
//  
//  NodePtr G = oa::ops::new_node(TYPE_PLUS, A, B);
//  NodePtr H = oa::ops::new_node(TYPE_MULT, G, C);
//  NodePtr I = oa::ops::new_node(TYPE_DIVD, H, D);
//  NodePtr J = oa::ops::new_node(TYPE_MINUS, E, F);
//  NodePtr K = oa::ops::new_node(TYPE_PLUS, I, J);
//
//  oa::ops::gen_kernels_JIT(K, true);
//
//  A->display("A");
//  B->display("B");
//  C->display("C");
//  D->display("D");
//  E->display("E");
//  F->display("F");
//
//  ArrayPtr ans = oa::ops::eval(K);
//  ans->display("eval");
//}
//
//// need 6 mpi_process
//void test_set() {
//  // A
//  //ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6}, {6, 6, 6}, {1}, 1);
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD,  {18, 18, 18}, {1}, 1);
//  ap->display("======A======");
//  // sub1
//  Box box1(4, 6, 5, 12, 0, 0);
//  
//  // sub2
//  Box box2(8, 10, 6, 13, 0, 0);
//  ArrayPtr sub_ap2 = oa::funcs::subarray(ap, box2);
//  sub_ap2->display("====sub_2=====");
//
//  //oa::funcs::set(ap, box1, sub_ap2);
//  //ap->display("======after_set======");
//
//  oa::funcs::set(ap, box1, ap, box2);
//  ap->display("======after_set======");
//
//  oa::funcs::set_ref_const(ap, box1, 0);
//  ap->display("======after_set======");
//}
//
//void test_rep() {
//  ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {4, 4, 4}, 1);
//  ap->display("======A======");
//
//  NodePtr NN = oa::ops::new_node(ap);
//  ArrayPtr lap = oa::funcs::consts(MPI_COMM_SELF, {3, 1, 1}, 2, 0);
//  NodePtr NN2 = oa::ops::new_node(lap);
//
//
//  NodePtr NP = oa::ops::new_node(TYPE_REP, NN, NN2);
//  ArrayPtr repA = oa::ops::eval(NP);
//  repA->display("======after_rep======");
//}
//
//void test_g2l(){
//  ArrayPtr global = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 1);
//  global->display("======global======");
//  ArrayPtr local = oa::funcs::g2l(global);
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  //sleep(rank);
//  if(rank == 3)
//    local->display("======local======");
//}
//
//void test_l2g(){
//  ArrayPtr lap = oa::funcs::seqs(MPI_COMM_SELF, {6, 6, 6}, 1);
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  if(rank == 3)
//    lap->display("======local======");
//  ArrayPtr global = oa::funcs::l2g(lap);
//  global->display("======global======");
//}
//
//
//// sub(A) = B (MPI_COMM_SELF)
//void test_set_l2g() {
//  // A
//  ArrayPtr local = oa::funcs::seqs(MPI_COMM_SELF, {3, 3, 3}, 0);
//  ArrayPtr global = oa::funcs::ones(MPI_COMM_WORLD, {8, 8, 8}, 0);
//
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//  if(rank == 0)
//    local->display("======local======");
//  global->display("======global======");
//
//  if(rank == 0)
//    std::cout<<"now set ..................."<<std::endl;
//  Box box(2, 4, 3, 5, 4, 6);
//  oa::funcs::set_l2g(global, box, local);
//  global->display("======global======");
//}
//
////local_A (MPI_COMM_SELF)= sub(global_B)
//void test_set_g2l() {
//  ArrayPtr local = oa::funcs::zeros(MPI_COMM_SELF, {3, 3, 3}, 0);
//  ArrayPtr global = oa::funcs::seqs(MPI_COMM_WORLD, {8, 8, 8}, 0);
//
//  int rank;
//  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//  if(rank == 2)
//    local->display("======local======");
//
//  global->display("======global======");
//
//  if(rank == 0)
//    std::cout<<"now set ..................."<<std::endl;
//
//  Box sub_box(2, 4, 3, 5, 4, 6);
//  oa::funcs::set_g2l(local, sub_box, global);
//  if(rank == 2)
//    local->display("======local======");
//
//
//}
//
//void test_fusion_operator() {
//  /*
//    ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//    ap->display("A");
//    ap->get_partition()->set_stencil_type(STENCIL_BOX);
//    oa::internal::set_ghost_consts((int*)ap->get_buffer(), ap->local_shape(), 0, 2);
//    Shape S = ap->buffer_shape();
//  
//    ArrayPtr ans = oa::funcs::zeros(MPI_COMM_WORLD, {6, 6, 6}, 2);
//    vector<MPI_Request> reqs;
//  
//    oa::funcs::update_ghost_start(ap, reqs, -1);
//    oa::funcs::calc_inside(ans, ap, {2, 1, 0}, {1, 2, 1});
//    oa::funcs::update_ghost_end(reqs);
//    oa::funcs::calc_outside(ans, ap, {2, 1, 0}, {1, 2, 1});
//  
//    ans->display("ans = a[i-2,j-1,k]+a[i+1,j+2,k+1]");
//
//    int rk = ap->rank();
//    if (rk == 0) {
//    ArrayPtr test = oa::funcs::seqs(MPI_COMM_SELF, {6, 6, 6}, 2);
//    ArrayPtr test_ans = oa::funcs::zeros(MPI_COMM_SELF, {6, 6, 6}, 2);
//    //test->display("test");
//    test->get_partition()->set_stencil_type(STENCIL_BOX);
//    oa::internal::set_ghost_consts((int*)test->get_buffer(), test->local_shape(), 0, 2);
//    oa::funcs::calc_inside(test_ans, test, {0, 0, 0}, {0, 0, 0});
//    test_ans->display("test_ans");
//    }
//
//  */
//  /*int rk = ap->rank();
//  
//    vector<MPI_Request> reqs;
//    oa::funcs::
//    oa::funcs::update_ghost_start(ap, reqs, -1);
//    oa::funcs::update_ghost_end(reqs);
//
//
//    printf("=====%d======\n", rk);
//    oa::utils::print_data(ap->get_buffer(), sp, DATA_INT);
//
//  */}
//
//void test_op() {
//  // ArrayPtr ap = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ap->display("A");
//  // ap->get_partition()->set_stencil_type(STENCIL_BOX);
//  // oa::internal::set_ghost_consts((int*)ap->get_buffer(), ap->local_shape(), 0, 2);
//  
//  // NodePtr A = oa::ops::new_node(ap);
//  // NodePtr B;
//  // ArrayPtr ans;
//  // B = oa::ops::new_node(TYPE_AXF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AXF");
//
//  // B = oa::ops::new_node(TYPE_AXB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AXB");
//  
//  // B = oa::ops::new_node(TYPE_AYF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AYF");
//
//  // B = oa::ops::new_node(TYPE_AYB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AYB");
//  
//  // B = oa::ops::new_node(TYPE_AZF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AZF");
//  
//  // B = oa::ops::new_node(TYPE_AZB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("AZB");
//  
//  // B = oa::ops::new_node(TYPE_DXF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DXF");
//  
//  // B = oa::ops::new_node(TYPE_DXB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DXB");
//  
//  // B = oa::ops::new_node(TYPE_DYF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DYF");
//  
//  // B = oa::ops::new_node(TYPE_DYB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DYB");
//  
//  // B = oa::ops::new_node(TYPE_DZF, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DZF");
//  
//  // B = oa::ops::new_node(TYPE_DZB, A);
//  // ans = oa::ops::force_eval(B);
//  // ans->display("DZB");
//  
//}
//
//
void test_fusion_op_3d(int m, int n, int k, int i) {
  // ArrayPtr dx = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
  // ArrayPtr dy = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
  // ArrayPtr dz = oa::funcs::consts(MPI_COMM_WORLD, {1, 1}, {1, 1}, {3, 3}, 1, 2);
  // // ArrayPtr A = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
  
  // dx->set_pseudo(true);
  // dy->set_pseudo(true);
  // dz->set_pseudo(true);

  // dx->set_bitset("110");
  // dy->set_bitset("110");
  // dz->set_bitset("001");
  
  // // // Grid::global()->init_grid('C', NULL, NULL, NULL);
  // Grid::global()->init_grid('C', dx, dy, dz);

  // // 3d 
  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap2 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap3 = oa::funcs::ones(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap4 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap5 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap6 = oa::funcs::ones(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap7 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap8 = oa::funcs::ones(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap9 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
  ArrayPtr ap10 = oa::funcs::ones(MPI_COMM_WORLD, {m, n, k}, 2);
  
  NodePtr w = NODE(ap1);
  NodePtr q2 = NODE(ap2);
  NodePtr dt_3d = NODE(ap3);
  NodePtr u = NODE(ap4);
  NodePtr aam = NODE(ap5);
  NodePtr h_3d = NODE(ap6);
  NodePtr q2b = NODE(ap7);
  NodePtr dum_3d = NODE(ap8);
  NodePtr v = NODE(ap9);
  NodePtr dvm_3d = NODE(ap10);
  NodePtr dhf = NODE(ap10);
  NodePtr dhb = NODE(ap10);
  NodePtr dti2 = oa::ops::new_seqs_scalar_node(1);
  
  ArrayPtr gg;
  
  NodePtr a1 = MULT(w, q2);
  NodePtr a2 = AZF(a1);
  NodePtr a3 = DZF(a2);
  NodePtr a4 = AXB(q2);
  NodePtr a5 = AXB(dt_3d);
  NodePtr a6 = MULT(a4, a5);
  NodePtr a7 = AZB(u);
  NodePtr a8 = MULT(a6, a7);
  NodePtr a9 = AXB(aam);
  NodePtr a10 = AZB(a9);
  NodePtr a11 = AXB(h_3d);
  NodePtr a12 = MULT(a10, a11);
  NodePtr a13 = DXB(q2b);
  NodePtr a14 = MULT(a12, a13);
  NodePtr a15 = MULT(a14, dum_3d);
  NodePtr a16 = MINUS(a8, a15);
  NodePtr a17 = DXF(a16);
  NodePtr axx = UMINUS(a3);
  NodePtr a18 = PLUS(axx, a17);
  NodePtr a19 = AYB(q2);
  NodePtr a20 = AYB(dt_3d);
  NodePtr a21 = MULT(a19, a20);
  NodePtr a22 = AZB(v);
  NodePtr a23 = MULT(a21, a22);
  NodePtr a24 = AYB(aam);
  NodePtr a25 = AZB(a24);
  NodePtr a26 = AYB(h_3d);
  NodePtr a27 = MULT(a25, a26);
  NodePtr a28 = DYB(q2b);
  NodePtr a29 = MULT(a27, a28);
  NodePtr a30 = MULT(a29, dvm_3d);
  NodePtr a31 = MINUS(a23, a30);
  NodePtr a32 = DYF(a31);
  NodePtr a33 = PLUS(a18, a32);
  NodePtr a34 = MULT(a33, dti2);
  NodePtr a35 = MULT(q2b, dhb);
  NodePtr a36 = MINUS(a35, a34);
  NodePtr a37 = DIVD(a36, dhf);
  
  //oa::ops::gen_kernels_JIT(a33);
  //ArrayPtr gao = EVAL(a33);
  //gao->display("gao");
  
  oa::ops::write_graph(a37);  
  
  


  // q2f= DZB(AZF(w*q2)) 
  // + DXF(AXB(q2)*AXB(dt_3d)*AZB(u) - AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)
  // + DYF(AYB(q2)*AYB(dt_3d)*AZB(v) - AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d)

}
//
//void test_fusion_op_2d(int m, int n, int k) {
//  // ArrayPtr dx = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  // ArrayPtr dy = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  // ArrayPtr dz = oa::funcs::consts(MPI_COMM_WORLD, {1, 1}, {1, 1}, {3, 3}, 1, 2);
//  // ArrayPtr A = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  
//  // dx->set_pseudo(true);
//  // dy->set_pseudo(true);
//  // dz->set_pseudo(true);
//
//  // NodePtr n1 = NODE(dx);
//  // NodePtr n2 = NODE(dy);
//  // NodePtr n3 = NODE(dz);
//  // NodePtr n4 = NODE(A);
//
//  // NodePtr n5 = PLUS(n4, n1);
//  // NodePtr n6 = PLUS(n4, n3);
//  // NodePtr n7 = PLUS(n2, n3);
//
//  // ArrayPtr ans1 = EVAL(n5);
//  // ArrayPtr ans2 = EVAL(n6);
//  // ArrayPtr ans3 = EVAL(n7);
//
//  // // Grid::global()->init_grid('C', NULL, NULL, NULL);
//  // Grid::global()->init_grid('C', dx, dy, dz);
//
//  
//  // // 3d 
//  // ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap2 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap3 = oa::funcs::ones(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap4 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap5 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap6 = oa::funcs::ones(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap7 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap8 = oa::funcs::ones(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap9 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ArrayPtr ap10 = oa::funcs::ones(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  
//  // pseudo 3d
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap2 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap3 = oa::funcs::consts(MPI_COMM_WORLD, {m/2, m/2}, {n/2, n/2}, {1, 1}, 1, 2);
//  ap3->set_pseudo(true);
//  ap3->set_bitset("110");
//  ArrayPtr ap4 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap5 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap6 = oa::funcs::consts(MPI_COMM_WORLD, {m/2, m/2}, {n/2, n/2}, {1, 1}, 1, 2);
//  ap6->set_pseudo(true);
//  ap6->set_bitset("110");
//  ArrayPtr ap7 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap8 = oa::funcs::consts(MPI_COMM_WORLD, {m/2, m/2}, {n/2, n/2}, {1, 1}, 1, 2);
//  ap8->set_pseudo(true);
//  ap8->set_bitset("110");
//  ArrayPtr ap9 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap10 = oa::funcs::consts(MPI_COMM_WORLD, {m/2, m/2}, {n/2, n/2}, {1, 1}, 1, 2);
//  ap10->set_pseudo(true);
//  ap10->set_bitset("110");
//
//  
//  NodePtr w = NODE(ap1);
//  NodePtr q2 = NODE(ap2);
//  NodePtr dt_3d = NODE(ap3);
//  NodePtr u = NODE(ap4);
//  NodePtr aam = NODE(ap5);
//  NodePtr h_3d = NODE(ap6);
//  NodePtr q2b = NODE(ap7);
//  NodePtr dum_3d = NODE(ap8);
//  NodePtr v = NODE(ap9);
//  NodePtr dvm_3d = NODE(ap10);
//  
//  ArrayPtr gg;
//  
//  NodePtr a1 = MULT(w, q2);
//  NodePtr a2 = AZF(a1);
//  NodePtr a3 = DZF(a2);
//  NodePtr a4 = AXB(q2);
//  NodePtr a5 = AXB(dt_3d);
//  NodePtr a6 = MULT(a4, a5);
//  NodePtr a7 = AZB(u);
//  NodePtr a8 = MULT(a6, a7);
//  NodePtr a9 = AXB(aam);
//  NodePtr a10 = AZB(a9);
//  NodePtr a11 = AXB(h_3d);
//  NodePtr a12 = MULT(a10, a11);
//  NodePtr a13 = DXB(q2b);
//  NodePtr a14 = MULT(a12, a13);
//  NodePtr a15 = MULT(a14, dum_3d);
//  NodePtr a16 = MINUS(a8, a15);
//  NodePtr a17 = DXF(a16);
//  NodePtr a18 = PLUS(a3, a17);
//  NodePtr a19 = AYB(q2);
//  NodePtr a20 = AYB(dt_3d);
//  NodePtr a21 = MULT(a19, a20);
//  NodePtr a22 = AZB(v);
//  NodePtr a23 = MULT(a21, a22);
//  NodePtr a24 = AYB(aam);
//  NodePtr a25 = AZB(a24);
//  NodePtr a26 = AYB(h_3d);
//  NodePtr a27 = MULT(a25, a26);
//  NodePtr a28 = DYB(q2b);
//  NodePtr a29 = MULT(a27, a28);
//  NodePtr a30 = MULT(a29, dvm_3d);
//  NodePtr a31 = MINUS(a23, a30);
//  NodePtr a32 = DYF(a31);
//  NodePtr a33 = PLUS(a18, a32);
//  
//  ArrayPtr gao = EVAL(a33);
//  gao->display("gao");
//  
//  //oa::ops::write_graph(a33);  
//  
//  //oa::ops::gen_kernels(a33);
//
//
//
//  // q2f= DZB(AZF(w*q2)) 
//  // + DXF(AXB(q2)*AXB(dt_3d)*AZB(u) - AZB(AXB(aam))*AXB(h_3d)*DXB(q2b)*dum_3d)
//  // + DYF(AYB(q2)*AYB(dt_3d)*AZB(v) - AZB(AYB(aam))*AYB(h_3d)*DYB(q2b)*dvm_3d)
//
//}
//
//void test_pseudo_3d() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  ArrayPtr ap2 = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  
//  ap2->set_pseudo(true);
//  
//  ap1->display("3d");
//  ap2->display("pseudo_3d");
//
//  NodePtr a1 = NODE(ap1);
//  NodePtr a2 = NODE(ap2);
//  NodePtr a3 = PLUS(a1, a2);
//
//  ArrayPtr ans = EVAL(a3);
//
//  ans->display("ans");
//
//
//}
//
//void test_rand() {
//  ArrayPtr ap = oa::funcs::rands(MPI_COMM_WORLD, {4, 4, 4}, 1, 1);
//  ap->display("rand");
//}
//
//void test_bitset() {
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  ArrayPtr ap2 = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  ap1->set_bitset("111");
//  ap2->set_bitset("110");
//  ap1->set_pseudo(true);
//  ap2->set_pseudo(true);
//  ap1->display("ap1");
//  ap2->display("ap2");
//
//  NodePtr n1 = NODE(ap1);
//  NodePtr n2 = NODE(ap2);
//  NodePtr n3 = MULT(n1, n2);
//  n1->display("n1");
//  n2->display("n2");
//  n3->display("n3");
//  
//  ArrayPtr ans = EVAL(n3);
//  ans->display("ans");
//
//}
//
//void test_operator_with_grid() {
//  ArrayPtr dx = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  ArrayPtr dy = oa::funcs::consts(MPI_COMM_WORLD, {3, 3}, {3, 3}, {1, 1}, 1, 2);
//  ArrayPtr dz = oa::funcs::consts(MPI_COMM_WORLD, {1, 1}, {1, 1}, {3, 3}, 1, 2);
//  
//  dx->set_pseudo(true);
//  dy->set_pseudo(true);
//  dz->set_pseudo(true);
//
//  dx->set_bitset("110");
//  dy->set_bitset("110");
//  dz->set_bitset("001");
//  
//  Grid::global()->init_grid('C', dx, dy, dz);
//
//  Grid::global()->get_grid_dx(3)->display("dx[3]");
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  // ap1->display("ap1");
//  ap1->set_pos(3);
//  
//  NodePtr n1 = NODE(ap1);
//  // n1->display("n1");
//  NodePtr n2 = DXF(n1);
//  // n2->display("n2");
//
//  ArrayPtr ans = EVAL(n2);
//  ans->display("ans");
//}
//
//void test_cache(int m, int n, int k) {
//  ArrayPtr ans;
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap2 = oa::funcs::seqs(MPI_COMM_WORLD, {m/2, m/2}, {n/2, n/2}, {1, 1}, 2);
//  ap2->set_bitset("110");
//  ArrayPtr ap3 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap4 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap5 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap6 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap7 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ArrayPtr ap8 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  
//  // ArrayPtr ap3 = oa::funcs::seqs(MPI_COMM_WORLD, {6, 6, 6}, 2);
//  
//  std::string tmp_node_key = "";
//  NodePtr tmp_node = NULL;
//
//  NodePtr n1 = NodePool::global()->get_seqs_scalar(MPI_COMM_SELF, 1);
//  NodePtr n2 = NODE(ap2);
//  NodePtr n3 = NODE(ap3);
//  NodePtr n4 = NODE(ap4);
//  NodePtr n5 = NODE(ap5);
//  NodePtr n6 = NODE(ap6);
//  NodePtr n7 = NODE(ap7);
//  NodePtr n8 = NODE(ap8);
//  
//  NodePtr np;
//
//  for (int i = 0; i < 10; i++) {
//    // CSET(np, PLUS( PLUS(n1, n2), PLUS(n3, n4) ) );
//    // oa::ops::gen_kernels_JIT(np);
//    // np = PLUS(PLUS(n1, n2), PLUS(n3, n4));
//    // tmp_node_key = gen_node_key(__FILE__, __LINE__);  
//    // find_node(tmp_node, tmp_node_key);                  
//    // if (is_valid(tmp_node)) {                           
//    //   np = tmp_node;                                     
//    // } else {
//    //   tmp_node = PLUS( PLUS(n1, n2), PLUS(n3, n4));
//    //   cache_node(tmp_node, tmp_node_key);               
//    //   np = tmp_node;                                     
//    // }
//    // ans = EVAL(np);
//  }
//
//  np = PLUS( PLUS(n1, n2), PLUS(n3, n4) );
//  oa::ops::gen_kernels_JIT_with_op(np);
//
//  // // tmp_node.reset();
//  // tmp_node = NULL;
//  // //ans->display("ans");
//
//  // NodePtr ND = PLUS(n1, n2);
//
//  // for (int i = 0; i < 10; i++) {
//  //   np = ND;
//  // }
//
//
//
//}
//
//void test_gen_kernel_JIT_with_op(int m, int n, int k) {
//  m = n = k = 6;
//
//  ArrayPtr dx = oa::funcs::consts(MPI_COMM_WORLD, {6, 6, 1}, 1, 2);
//  ArrayPtr dy = oa::funcs::consts(MPI_COMM_WORLD, {6, 6, 1}, 1, 2);
//  ArrayPtr dz = oa::funcs::consts(MPI_COMM_WORLD, {1, 1, 6}, 1, 2);
//  
//  dx->set_pseudo(true);
//  dy->set_pseudo(true);
//  dz->set_pseudo(true);
//
//  dx->set_bitset("110");
//  dy->set_bitset("110");
//  dz->set_bitset("001");
//  
//  Grid::global()->init_grid('C', dx, dy, dz);
//
//
//  ArrayPtr ans;
//  ArrayPtr ap1 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, k}, 2);
//  ap1->set_bitset("111");
//  ap1->set_pos(3);
//  ArrayPtr ap2 = oa::funcs::seqs(MPI_COMM_WORLD, {m, n, 1}, 2);
//  ap2->set_bitset("110");
//  ap2->set_pos(2);
//  ap2->set_pseudo(true);
//
//  
//  NodePtr n1 = NODE(ap1);
//  NodePtr n2 = NODE(ap2);
//  n1->display("n1");
//  n2->display("n2");
//  NodePtr n3 = NodePool::global()->get_seqs_scalar(MPI_COMM_SELF, 1);
//  
//  NodePtr np = PLUS( MULT(DXF(n1), DYB(n2)), n3 );
//  
//  // ans = oa::ops::eval(np);
//  // ans->display("force eval");
//
//  // oa::ops::gen_kernels_JIT(np);
//  // ans = oa::ops::eval(np);
//  // ans->display("JIT eval");
//
//  oa::ops::gen_kernels_JIT_with_op(np);
//  ans = oa::ops::eval_with_op(np);
//  ans->display("JIT eval with op");
//
//}
void test_set_with_mask() {
  // A

  ArrayPtr A = oa::funcs::zeros(MPI_COMM_WORLD,  {6, 6, 6}, {1}, 1);
  ArrayPtr B = oa::funcs::consts(MPI_COMM_WORLD,  {6, 6, 6}, 199, 1);
  //ArrayPtr B = oa::funcs::consts(MPI_COMM_WORLD,  {1, 1, 1}, 199, 1);
  ArrayPtr mask = oa::funcs::seqs(MPI_COMM_WORLD,  {6, 6, 6}, {1}, 1);

  A->display("======A======");
  B->display("====B====");
  mask->display("====mask====");

  //oa::funcs::set(A, box1, sub_ap);
  //oa::funcs::set_with_mask(A, box1, sub_ap,sub_ap);
  oa::funcs::set_with_mask(A, B, mask);
  A->display("======after_set======");

}

#endif
