#ifndef __TEST_HPP__
#define __TEST_HPP__

#include "../Range.hpp"
#include "../Box.hpp"
#include "../Partition.hpp"
#include "../Array.hpp"
#include "../Function.hpp"

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
	/*
	ap = oa::funcs::consts(MPI_COMM_WORLD, {2, 2}, {2, 2}, {4}, 1, 1);
	if (rank == 0) ap->get_partition()->display("Array_Consts_m2");

	ap = oa::funcs::ones(MPI_COMM_WORLD, {4, 4, 4}, 1);
	if (rank == 0) ap->get_partition()->display("Array_ones");
	
	ap = oa::funcs::zeros(MPI_COMM_WORLD, {4, 4, 4}, 1);
	if (rank == 0) ap->get_partition()->display("Array_zeros");	
	*/
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

#endif
