#include "Operator.hpp"
#include "Grid.hpp"
#include <boost/throw_exception.hpp>

#define AXB(x) oa::ops::new_node(TYPE_AXB, x)
#define AYB(x) oa::ops::new_node(TYPE_AYB, x)
#define AZB(x) oa::ops::new_node(TYPE_AZB, x)
#define EVAL(x) oa::ops::eval(x)

void Grid::init_grid(char type, ArrayPtr dx, ArrayPtr dy, ArrayPtr dz){

  NodePtr ndx = oa::ops::new_node(dx);
  NodePtr ndy = oa::ops::new_node(dy);
  NodePtr ndz = oa::ops::new_node(dz);

  switch(type){
  case 'C':
    x_d[0] = EVAL(AYB(AXB(ndx)));
    y_d[0] = EVAL(AYB(AXB(ndy)));
    z_d[0] = dz;

    x_d[1] = EVAL(AYB(ndx));
    y_d[1] = EVAL(AYB(ndy));
    z_d[1] = dz;

    x_d[2] = EVAL(AXB(ndx));
    y_d[2] = EVAL(AXB(ndy));
    z_d[2] = dz;

    x_d[3] = dx;
    y_d[3] = dy;
    z_d[3] = dz;

    x_d[4] = EVAL(AYB(AXB(ndx)));
    y_d[4] = EVAL(AYB(AXB(ndy)));
    z_d[4] = EVAL(AZB(ndz));

    x_d[5] = EVAL(AYB(ndx));
    y_d[5] = EVAL(AYB(ndy));
    z_d[5] = EVAL(AZB(ndz));

    x_d[6] = EVAL(AXB(ndx));
    x_d[6] = EVAL(AXB(ndy));
    x_d[6] = EVAL(AZB(ndz));

    x_d[7] = dx;
    y_d[7] = dy;
    z_d[7] = EVAL(AZB(ndz));

    x_map.resize(8);
    y_map.resize(8);
    z_map.resize(8);

    x_map[0] = 1;  y_map[0] = 2;  z_map[0] = 4;
    x_map[1] = 0;  y_map[1] = 3;  z_map[1] = 5;
    x_map[2] = 3;  y_map[2] = 0;  z_map[2] = 6;
    x_map[3] = 2;  y_map[3] = 1;  z_map[3] = 7;
    x_map[4] = 5;  y_map[4] = 6;  z_map[4] = 0;
    x_map[5] = 4;  y_map[5] = 7;  z_map[5] = 1;    
    x_map[6] = 7;  y_map[6] = 4;  z_map[6] = 2;
    x_map[7] = 6;  y_map[7] = 5;  z_map[7] = 3;
    
    break;
  default:
    BOOST_THROW_EXCEPTION(std::logic_error("unsupported grid type"));
    break;
  }
}

ArrayPtr Grid::get_grid_dx(int pos){
  return x_d[pos];
}

ArrayPtr Grid::get_grid_dy(int pos){
  return y_d[pos];
}

ArrayPtr Grid::get_grid_dz(int pos){
  return z_d[pos];
}
