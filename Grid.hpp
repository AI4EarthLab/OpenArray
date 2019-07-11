/*
 * Grid.hpp
 * currently, OpenArray only support C grid
 *
=======================================================*/

#ifndef __GRID_HPP__
#define __GRID_HPP__

#include "Array.hpp"
#include <unordered_map>

class Grid {
  private:
  char grid_type;

  // grid map
  std::unordered_map<int, ArrayPtr> x_d;
  std::unordered_map<int, ArrayPtr> y_d;
  std::unordered_map<int, ArrayPtr> z_d;

  // 
  vector<int> x_map;
  vector<int> y_map;
  vector<int> z_map;
  
  public:
  // dx, dy, dz are two dimensional arrays
  void init_grid(char type, const ArrayPtr& dx,
          const ArrayPtr& dy, const ArrayPtr& dz);

  // get x/y/z dimension grid based on grid position
  ArrayPtr get_grid_dx(int pos);
  ArrayPtr get_grid_dy(int pos);
  ArrayPtr get_grid_dz(int pos);

  // get grid based on grid position and Operator type
  ArrayPtr get_grid(int pos, NodeType t);

  // get grid position based on grid position and Operator type
  int get_pos(int pos, NodeType t);
    
  int get_pos_x(int pos);
  int get_pos_y(int pos);
  int get_pos_z(int pos);

  static Grid* global();
};

typedef std::shared_ptr<Grid> GridPtr;

#endif
