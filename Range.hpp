/*
 * Range.hpp
 * Range [lower, upper)
 *
=======================================================*/

#ifndef __RANGE_HPP__
#define __RANGE_HPP__

#include <iostream>

class Range {
  private:
  int m_lower;    // lower range
  int m_upper;    // upper range

  public:
  // Constructor
  Range();
  Range(int st, int ed);
  
  // get methods
  int get_lower() const;
  int get_upper() const;

  // check two range is equal or not
  bool equal(int st, int ed) const;
  bool equal(const Range &rg) const;

  // get range's size
  int size() const;
  
  // display range with prefix string(exp. name)
  void display(char const *prefix = "") const;

  // check if Range is inside u
  bool is_inside(const Range &u);

  // [lower, upper) + num = [lower + num, upper + num)
  void shift(int num);

  // check if Range has intersection with u
  bool intersection(const Range &u); 

  // get intersection Range with u
  Range get_intersection(const Range &u);

};

#endif
