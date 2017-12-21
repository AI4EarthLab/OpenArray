#ifndef __BOX_HPP__
#define __BOX_HPP__

#include <vector>
#include <memory>
#include "common.hpp"
#include "Range.hpp"

/*
 * Box:
 *  dimension: rx, ry, rz
 */

class Box {
  private:
  Range m_rx, m_ry, m_rz;

  public:
  Box();
  Box(Range x, Range y, Range z);
  Box(int* starts, int* counts);
  Box(int sx, int ex, int sy, int ey, int sz, int ez);
  
  Range get_range_x();
  Range get_range_y();
  Range get_range_z();
  void get_corners(int &xs, int &xe, int &ys, int &ye, int &zs, int &ze, int sw = 0) const;

  bool equal(const Box &u);
  bool equal_shape(const Box &b);
  void display(char const *prefix = "") const;

  // check if Box is inside Box u
  bool is_inside(const Box &u);

  // check if Box and Box u has intersection
  bool intersection(const Box &u);

  // get the intersection box
  Box get_intersection(const Box &u);

  // get the intersection box
  Box get_intersection1(const Box &u);
  
  Box ref_box(const Box &u) const;
  
  Box boundary_box(int sw) const;

  // return [shape_x, shape_y, shape_z]
  Shape shape(int sw = 0) const;

  // return shape_x * shape_y * shape_z
  int size(int stencil_width = 0);

  int3 starts() const;
  int3 counts() const;

  Box shift(int i);

};

typedef std::shared_ptr<Box> BoxPtr;

#endif
