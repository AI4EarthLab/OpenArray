#ifndef __GRID_HPP__
#define __GRID_HPP__

#include "Array.hpp"

class Grid {
  private:
  char grid_type;

  std::unordered_map<int, ArrayPtr> x_d;
  std::unordered_map<int, ArrayPtr> y_d;
  std::unordered_map<int, ArrayPtr> z_d;

  vector<int> x_map;
  vector<int> y_map;
  vector<int> z_map;
  
  public:
  //dx, dy, dz are two dimensional arrays
  void init_grid(char type, ArrayPtr dx, ArrayPtr dy, ArrayPtr dz);
  ArrayPtr get_grid_dx(int pos);
  ArrayPtr get_grid_dy(int pos);
  ArrayPtr get_grid_dz(int pos);

  int get_pos(int pos, NodeType t);
    
  int get_pos_x(int pos);
  int get_pos_y(int pos);
  int get_pos_z(int pos);

  static Grid* global();
};

typedef std::shared_ptr<Grid> GridPtr;

#endif
