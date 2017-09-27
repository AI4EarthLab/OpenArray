

#ifndef __ARRAYPOOL_HPP__
#define __ARRAYPOOL_HPP__

#include "Array.hpp"
#include <unordered_map>
#include <list>
#include <vector>

//Array pool
class ArrayPool{
  typedef std::unordered_map<size_t, ArrayList*> ArrayPoolMap;  
private:
  ArrayPoolMap m_pools;
public:
  
  ArrayPtr get(PartitionPtr& p){

    Array* ap;

    size_t ap_hash = size_t(p.get());
    
    ArrayPoolMap::iterator it = m_pools.find(ap_hash);

    if(it == m_pools.end()){
      ap = new Array(p);  //not found ArrayPool  
    }
    else if(it->second->size() < 1){
      ap = new Array(p);  //ArrayPool is empty
    }else{
      ap = it->second->back();
      it->second->pop_back();
    }
    
    return ArrayPtr(ap,
		    [](Array* arr_p){
		      ArrayPool::global()->dispose(arr_p); 
		    });
  }

  ArrayPtr get(int shape[3], int pshape[3],
	       MPI_Comm comm = MPI_COMM_WORLD,
	       BoundType bound_type[3]={0, 0, 0},
	       StencilType stencil_type[3]={0,0,0},
	       int stencil_width=0){
    
  }

  ArrayPtr get(std::vector<int>& lx,
	       std::vector<int>& ly,
	       std::vector<int>& lz,
	       MPI_Comm comm = MPI_COMM_WORLD,
	       BoundType bound_type[3]={0, 0, 0},
	       StencilType stencil_type[3]={0,0,0},
	       int stencil_width=0){
    
  }

  ArrayPtr get(int shape[3],
	       MPI_Comm comm = MPI_COMM_WORLD){
    
  }
  
  void dispose(Array* arr){

    std::cout<<"dispose called!\n"<<std::endl;
    
    size_t arr_hash = arr->struct_hash();
    
    ArrayPoolMap::iterator it = m_pools.find(arr_hash);
    
    if(it == m_pools.end()){
      ArrayList* la = new ArrayList();
      la->push_back(arr);
      m_pools[arr_hash] = la;
    }else{
      it->second->push_back(arr);      
    }
  }
  
  static ArrayPool* global(){
    static ArrayPool ap;
    return &ap;
  }
};


#endif
