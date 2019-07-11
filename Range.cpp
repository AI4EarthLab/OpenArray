/*
 * Range.cpp
 * Range [lower, upper)
 *
=======================================================*/

#include<iostream>
#include "Range.hpp"
#include  <algorithm>

using namespace std;

Range::Range() {
  m_lower = m_upper = 0;  
}

Range::Range(int st, int ed) : 
  m_lower(st), m_upper(ed) {}

int Range::get_lower() const {
  return m_lower;
}

int Range::get_upper() const {
  return m_upper;
}

bool Range::equal(int st, int ed) const {
  return m_lower == st && m_upper == ed;
}

bool Range::equal(const Range &rg) const {
   return m_lower == rg.m_lower && m_upper == rg.m_upper; 
}

int Range::size() const{
  return std::max(m_upper - m_lower, 0);
}

void Range::display(char const *prefix) const {
  printf("Range %s is [%d, %d)\n", prefix, m_lower, m_upper);
}

bool Range::is_inside(const Range &rg) {
  return rg.m_lower <= m_lower && m_upper <= rg.m_upper;
}

void Range::shift(int num) {
  m_lower += num;
  m_upper += num;
}

bool Range::intersection(const Range &u) {
  if (m_lower >= u.m_upper || u.m_lower >= m_upper) return false;
  return true;
}

Range Range::get_intersection(const Range &u) {
  if (!intersection(u)) return Range();
  return Range(max(m_lower, u.m_lower), min(m_upper, u.m_upper));
}

