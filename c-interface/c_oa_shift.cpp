
#include "../NodePool.hpp"

extern "C"{
  void c_oa_shift(NodePtr*& A, NodePtr*& B){

    
    //ArrayPtr* p = &(*A);
    std::cout<<"Heeeeeee1"<<std::endl;
    // oa::funcs::translate(*(ArrayPtr*)A, dir, dis, (bool)rou);
    std::cout<<"Heeeeeee2"<<std::endl;
    //A = p;
  }
}

