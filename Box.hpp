/*
 * Box:
 *  three dimension: rx, ry, rz
 *
=======================================================*/

#ifndef __BOX_HPP__
#define __BOX_HPP__

#include <vector>
#include <memory>
#include "Range.hpp"
#include "common.hpp"

class Box {
  private:
  Range m_rx, m_ry, m_rz;   // three dimension range

  public:
  // Constructor
  Box();
  Box(Range x, Range y, Range z);
  Box(int* starts, int* counts);
  Box(int sx, int ex, int sy, int ey, int sz, int ez);

  int xs() const;  // get dimension x's lower
  int xe() const;  // get dimension x's upper
  int ys() const;  // get dimension y's lower
  int ye() const;  // get dimension y's upper
  int zs() const;  // get dimension z's lower
  int ze() const;  // get dimension z's upper
  
  Range get_range_x();  // get dimension x's range
  Range get_range_y();  // get dimension y's range
  Range get_range_z();  // get dimension z's range

  // get Box's corners {[xs, xe), [ys, ye), [zs, ze)}
  void get_corners(
      int &xs, int &xe, int &ys, int &ye, int &zs, int &ze) const;

  // get Box's corners with stencil {[xs, xe), [ys, ye), [zs, ze)}
  void get_corners_with_stencil(
      int &xs, int &xe, int &ys, int &ye, int &zs, int &ze, int sw = 0) const;

  // check if Box is equal with u
  bool equal(const Box &u);

  // check if Box's shape is equal with u
  bool equal_shape(const Box &u);
  
  // display Box's information with prefix string
  void display(char const *prefix = "") const;

  // check if Box is inside Box u
  bool is_inside(const Box &u);

  // check if Box and Box u has intersection
  bool intersection(const Box &u);

  // get the intersection box
  Box get_intersection(const Box &u);
  
  // get the ref new box based on reference box u
  Box ref_box(const Box &u) const;
  
  // get the boundary box which include the stencil
  Box boundary_box(int sw) const;

  // return [shape_x, shape_y, shape_z]
  Shape shape() const;

  // return [shape_x, shape_y, shape_z] with the stencil
  Shape shape_with_stencil(int sw = 0) const;

  // return shape_x * shape_y * shape_z
  int size() const;

  // return shape_x * shape_y * shape_z with the stencil
  int size_with_stencil(int stencil_width = 0) const;

  // return box starts [rx.lower, ry.lower, rz.lower]
  oa_int3 starts() const;

  // return box sizes [rx.size, ry.size, rz.size]
  oa_int3 counts() const;

  // shift the box with i
  Box shift(int i);

  // shift the box with [x, y, z]
  Box shift(int x, int y, int z);
  
};

typedef std::shared_ptr<Box> BoxPtr;

#endif
