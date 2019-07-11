#include "NodeVec.hpp"
#include "Simple_Node.hpp"
#include <iostream>
using namespace std;

//void NodeVec::put_datanode_s(Simple_node& np_s)
//{
//    //ndptr_vec_s.push_back(np_s);
//    datand_vec_s.push_back(np_s);
//}


//void NodeVec::put_opnode_s(Simple_node& np_s)
//{
//	ndptr_vec_s.push_back(np_s);
//}

//sqx modify
int NodeVec::get_datanode_size() 
{
	return datanode_size_real;
}

int NodeVec::get_opnode_size()
{
	return opnode_size_real;
}

void NodeVec::set_nodesize_zero()
{
	datanode_size_real = 0;
	opnode_size_real = 0;
}

void NodeVec::set_datanode_size() 
{
	datanode_size_real++;
	opnode_size_real++;
}

void NodeVec::set_opnode_size()
{
	opnode_size_real++;
}

int NodeVec::get_nodevec_size()
{
	return ndptr_vec_s.size();
}

int NodeVec::get_datavec_size()
{
	return datand_vec_s.size();
}

//void NodeVec::set_datavec(Simple_node np_s)
//{
//	datand_vec_s[datanode_size_real-1] = np_s;
//}

void NodeVec::set_datavec(int type, extra_info info)
{	
	datand_vec_s[datanode_size_real-1].set_type(type);
	datand_vec_s[datanode_size_real-1].set_info(info);
	datand_vec_s[datanode_size_real-1].set_ArrayPtr_NULL();
}

void NodeVec::set_datavec(int type, ArrayPtr* &ap)
{	
	datand_vec_s[datanode_size_real-1].set_type(type);
	datand_vec_s[datanode_size_real-1].set_ArrayPtr(ap);
}

void NodeVec::set_datavec(int type, int *id1, extra_info info)
{
	datand_vec_s[datanode_size_real-1].set_type(type);
	datand_vec_s[datanode_size_real-1].set_info(info);
	datand_vec_s[datanode_size_real-1].input[0] = *id1;
	datand_vec_s[datanode_size_real-1].set_ArrayPtr_NULL();
}

//void NodeVec::set_nodevec(Simple_node np_s)
//{
//	ndptr_vec_s[opnode_size_real-1] = np_s;
//}

void NodeVec::set_nodevec(int type, int *id1, extra_info info)
{
	ndptr_vec_s[opnode_size_real-1].set_type(type);
	ndptr_vec_s[opnode_size_real-1].input[0] = *id1;
	ndptr_vec_s[opnode_size_real-1].set_info(info);
	ndptr_vec_s[opnode_size_real-1].set_ArrayPtr_NULL();
}

void NodeVec::set_nodevec(int type, int *id1, int *id2)
{
	ndptr_vec_s[opnode_size_real-1].set_type(type);
	ndptr_vec_s[opnode_size_real-1].input[0] = *id1;
	ndptr_vec_s[opnode_size_real-1].input[1] = *id2;
	ndptr_vec_s[opnode_size_real-1].set_ArrayPtr_NULL();
}

void NodeVec::set_nodevec(int type, int *id1)
{
	ndptr_vec_s[opnode_size_real-1].set_type(type);
	ndptr_vec_s[opnode_size_real-1].input[0] = *id1;
	ndptr_vec_s[opnode_size_real-1].set_ArrayPtr_NULL();
}

void NodeVec::set_nodevec(int type, extra_info info)
{	
	ndptr_vec_s[opnode_size_real-1].set_type(type);
	ndptr_vec_s[opnode_size_real-1].set_info(info);
	ndptr_vec_s[opnode_size_real-1].set_ArrayPtr_NULL();
}

void NodeVec::set_nodevec(int type, ArrayPtr* &ap)
{
	ndptr_vec_s[opnode_size_real-1].set_type(type);
	ndptr_vec_s[opnode_size_real-1].set_ArrayPtr(ap);
}

void NodeVec::reset_nodevec() {
	ndptr_vec_s.resize(opnode_size_real*2);
}

void NodeVec::reset_datavec() {
	datand_vec_s.resize(datanode_size_real*2);
}

size_t NodeVec::get_hash()
{
	size_t hash = 0;
	size_t seed = 13131; // 31 131 1313 13131 131313 etc..
	//int nodesize = ndptr_vec_s.size();
	int nodesize = opnode_size_real;
	for(int i = 0; i < nodesize; ++i)
	{
		auto& cur_node = ndptr_vec_s[i];
		if(cur_node.input[0]!=-1){
			hash = hash*seed + cur_node.input[0];
		}
		if(cur_node.input[1]!=-1){
			hash = hash*seed + cur_node.input[1];
		}
		int node_type = cur_node.type;
		if (TYPE_DATA == node_type) {
			hash = hash *seed + cur_node.get_ArrayPtr()->get_pos();
			hash = hash*seed + cur_node.get_ArrayPtr()->get_hash();
		}
		//sub node
		else if(TYPE_REF == node_type){
			int *box = (int*)cur_node.get_val();
			int shape[3];
			shape[0] = box[1]-box[0];
			shape[1] = box[3]-box[2];
			shape[2] = box[5]-box[4];
			for(int j=0;j<3;j++){
				hash = hash*seed + shape[j];
			}      
		}
		//int3_rep
		else if(TYPE_INT3_REP == node_type){
			int *val = (int*)cur_node.get_val();
			for(int j=0;j<3;j++){
				hash = hash*seed + val[j];
			}
		}
		else{
			hash = hash*seed + node_type;
		}
	}
	return hash;
}

vector<Simple_node>& NodeVec::get_ndptr_s() 
{
	return ndptr_vec_s;
}

vector<Simple_node>& NodeVec::get_datand_s()
{
	return datand_vec_s;
}
void NodeVec::clear()
{
	ndptr_vec_s.clear();
	datand_vec_s.clear();
}
