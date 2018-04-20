/*
 * not used currently
 *
=======================================================*/

#ifndef __BOUNDARYPOOL_HPP__
#define __BOUNDARYPOOL_HPP__

#include "Boudary.hpp"
#include "common.hpp"
#include <unordered_map>
#include <list>

using namespace std;

typedef list<Boundary*> BoundaryList;
typedef unordered_map<int, BoundaryList*> BoundaryPoolMap;

class BoundaryPool {
private:
  BoundaryPoolMap m_pools;

public:
  BoundaryPtr get(const Shape& ls, int sw) {
    Boundary* bp;
    int mx_length = max(max(ls[0], ls[1]), ls[2]) + 2 * sw;
    int size = mx_length * mx_length * sw * 6;

    BoundaryPoolMap::iterator it = m_pools.find(size);

    if (it == m_pools.end() || it->second->size() < 1) {
      bp = new Boundary(size);
    } else {
      bp = it->second->back();
      it->second->pop_back();
    }

    return BoundaryPtr(bp, [](Boundary* bd_p) {
      BoundaryPool::global()->dispose(bd_p);
    });
  }

  void dispose(Boundary* bp) {
    int size = bp->size();
    BoundaryPoolMap::iterator it = m_pools.find(size);

    if (it == m_pools.end()) {
      BoundaryList* bl = new BoundaryList();
      bl->push_back(bp);
      m_pools[size] = bl;
    } else {
      it->second->push_back(bp);
    }
  }

  static BoundaryPool* global() {
    static BoundaryPool bp;
    return &bp;
  }

};

#endif
