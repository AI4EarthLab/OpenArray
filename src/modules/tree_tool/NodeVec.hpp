#ifndef __NODEVEC_HPP__
#define __NODEVEC_HPP__

//#include "kernel.hpp"
#include <sstream>
#include "Simple_Node.hpp"
#include "../../NodeDesc.hpp"
#include "../../Function.hpp"
#include "../../op_define.hpp"


class NodeVec
{
private:
    /* data */
  //std::vector<Simple_node> ndptr_vec_s; 
  //std::vector<Simple_node> datand_vec_s; 
  vector<Simple_node> ndptr_vec_s; 
  vector<Simple_node> datand_vec_s; 
	int datanode_size_real;
	int opnode_size_real;
public:
  //NodeVec(/* args */);
	NodeVec() {
		datanode_size_real = 0;
		opnode_size_real = 0;
	}

	~NodeVec(){};
	void clear();
	//void put_datanode(NodePtr& np);
	//void put_datanode_s(Simple_node& np_s);
	//void put_opnode(NodePtr& np);
	//void put_opnode_s(Simple_node& np);
	//void put_dataidx(int idx);
	//int  get_nodesize();

	//sqx modify
	int get_datanode_size();
	int get_opnode_size();
	void set_nodesize_zero();
	void set_datanode_size();
	void set_opnode_size();

	int get_datavec_size();
	int get_nodevec_size();
	
  //void set_nodevec(Simple_node np_s);
  //void set_datavec(Simple_node np_s);
  void set_datavec(int type, extra_info info);
  void set_datavec(int type, ArrayPtr* &ap);
  void set_datavec(int type, int *id1, extra_info info);

  void set_nodevec(int type, int *id1, extra_info info);
  void set_nodevec(int type, int *id1, int *id2);
  void set_nodevec(int type, int *id1);
  void set_nodevec(int type, extra_info info);
  void set_nodevec(int type, ArrayPtr* &ap);

	void reset_nodevec();
	void reset_datavec();

	bool empty();
	size_t get_hash();

	int index(){
		//return ndptr_vec_s.size()-1;
		return opnode_size_real-1;
	}


	vector<Simple_node>& get_ndptr_s() ;
	vector<Simple_node>& get_datand_s() ;

	static NodeVec* global()
	{
		static NodeVec nv;
		return &nv;
	}
};

#endif
