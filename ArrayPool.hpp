#ifndef __ARRAYPOOL_HPP__
#define __ARRAYPOOL_HPP__

#include "common.hpp"
#include "Array.hpp"
#include "PartitionPool.hpp"
#include <unordered_map>
#include <list>
#include <vector>
#include "log.hpp"

using namespace std;

typedef list<Array*> ArrayList;
typedef unordered_map<size_t, ArrayList*> ArrayPoolMap; 

/*
 * Array Pool:
 *  m_pools:  use [Partition info, buffer_data_type] as identical key
 *  value is an pointer of Arraylist
 */

class ArrayPool{
private:
  ArrayPoolMap m_pools;
  int global_count = 0;
  
public:
  // get an ArrayPtr from m_pools based on hash created by key:
  // [comm, process_size, (gx, gy, gz), stencil_width, buffer_data_type]
  ArrayPtr get(MPI_Comm comm, const Shape& gs, int stencil_width = 1, 
               int data_type = DATA_DOUBLE) {
    //oa::logging::write_log_error(1, "get ArrayPtr from ArrayPool");
    
    Array* ap;
    size_t par_hash = Partition::gen_hash(comm, gs, stencil_width);
    
    // only use array_hash = par_hash + buffer_data_type (may be a bug)
    size_t array_hash = par_hash + data_type;

    ArrayPoolMap::iterator it = m_pools.find(array_hash);

    int size;
    MPI_Comm_size(comm, &size);

    //  not found in ArrayPool 
    // OR found, but arraylist is empty
    if (it == m_pools.end() || it->second->size() < 1) { 
      PartitionPtr par_ptr = PartitionPool::global()->
        get(comm, size, gs, stencil_width, par_hash);
      ap = new Array(par_ptr, data_type);
      add_count();
      if (g_debug) cout<<"ArrayPool.size() = "<<count()<<endl;
      ap->set_hash(array_hash);
      //cout<<"ones new array"<<endl;
    } else {
      ap = it->second->back();
      it->second->pop_back();
    }

    // set bitset based on array global shape [m, n, k]
    ap->set_bitset();

    // ArrayPtr constructor with (Array pointer, Del del)
    return ArrayPtr(ap, 
                    [](Array* arr_p) {
                      ArrayPool::global()->dispose(arr_p);
                    });
  }

  // get an ArrayPtr from m_pools based on hash created by key:
  // [comm, lx, ly, lz, stencil_width, buffer_data_type]
  ArrayPtr get(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
               const vector<int> &z, int stencil_width = 1, int data_type = DATA_DOUBLE) {
    Array* ap;
    size_t par_hash = Partition::gen_hash(comm, x, y, z, stencil_width);
    
    // only use array_hash = par_hash + buffer_data_type (may be a bug)
    size_t array_hash = par_hash + data_type;

    ArrayPoolMap::iterator it = m_pools.find(array_hash);
    
    //  not found in ArrayPool 
    // OR found, but arraylist is empty
    if (it == m_pools.end() || it->second->size() < 1) { 
      PartitionPtr par_ptr = PartitionPool::global()->
        get(comm, x, y, z, stencil_width, par_hash);
      ap = new Array(par_ptr, data_type);
      add_count();
      if (g_debug) cout<<"ArrayPool.size() = "<<count()<<endl;
      ap->set_hash(array_hash);
    } else {
      ap = it->second->back();
      it->second->pop_back();
    }

    // set bitset based on array global shape [m, n, k]
    ap->set_bitset();

    // ArrayPtr constructor with (Array pointer, Del del)
    return ArrayPtr(ap, 
                    [](Array* arr_p) {
                      ArrayPool::global()->dispose(arr_p);
                    });
  }

  // get an ArrayPtr from m_pools based on partitionptr pp
  ArrayPtr get(const PartitionPtr &pp, int data_type = DATA_DOUBLE) {
    Array* ap;
    size_t par_hash = pp->get_hash();
    size_t array_hash = par_hash + data_type;

    ArrayPoolMap::iterator it = m_pools.find(array_hash);

    //cout<<"array get called. hash : "<<array_hash<<endl;

    if (it == m_pools.end() || it->second->size() < 1) {
      ap = new Array(pp, data_type);
      add_count();
      if (g_debug) cout<<"ArrayPool.size() = "<<count()<<endl;
      ap->set_hash(array_hash);
      //cout<<"new array"<<endl;
    } else {
      ap = it->second->back();
      it->second->pop_back();
      //cout<<"array from list"<<endl;
    }

    // set bitset based on array global shape [m, n, k]
    ap->set_bitset();

    return ArrayPtr(ap, [](Array* arr_p) {
                      ArrayPool::global()->dispose(arr_p);
                    });
  }

  void dispose(Array* ap){
    size_t array_hash = ap->get_hash();
    //cout<<array_hash<<" ArrayPool dispose called!\n"<<endl;
    ap->reset();

    ArrayPoolMap::iterator it = m_pools.find(array_hash);
    if (it == m_pools.end()){
      ArrayList* al = new ArrayList();
      al->push_back(ap);
      m_pools[array_hash] = al;
    } else{
      it->second->push_back(ap);  
      //cout<<"arraylist has: "<<it->second->size()<<endl;
    }

    //printf("array disposed called. hash:%d\n", array_hash);
  }
  
  // only need one Array Pool in each process, make it static 
  static ArrayPool* global(){
    static ArrayPool ap;
    return &ap;
  }

  static ArrayPtr* global_Array() {
    static ArrayPtr ap;
    return &ap;
  }

  int count() {
    return global_count;
  }

  void add_count() {
    global_count += 1;
  }

};


#endif
