#include "Operator.hpp"
#include "Grid.hpp"
#include "op_define.hpp"
#include <boost/throw_exception.hpp>
#include "common.hpp"
#include "utils/utils.hpp"
#include "MPI.hpp"

#define PSU3D(x) oa::funcs::make_psudo3d(x)

void Grid::init_grid(char type,
        const ArrayPtr& dx, const ArrayPtr& dy, const ArrayPtr& dz){

  NodePtr ndx = oa::ops::new_node(dx);
  NodePtr ndy = oa::ops::new_node(dy);
  NodePtr ndz = oa::ops::new_node(dz);

  switch(type){
  case 'C':
    x_d[0] = PSU3D(EVAL(AYB(AXB(ndx))));
    y_d[0] = PSU3D(EVAL(AYB(AXB(ndy))));
    z_d[0] = PSU3D(dz);

    x_d[1] = PSU3D(EVAL(AYB(ndx)));
    y_d[1] = PSU3D(EVAL(AYB(ndy)));
    z_d[1] = PSU3D(dz);

    x_d[2] = PSU3D(EVAL(AXB(ndx)));
    y_d[2] = PSU3D(EVAL(AXB(ndy)));
    z_d[2] = PSU3D(dz);

    x_d[3] = PSU3D(dx);
    y_d[3] = PSU3D(dy);
    z_d[3] = PSU3D(dz);

    x_d[4] = PSU3D(EVAL(AYB(AXB(ndx))));
    y_d[4] = PSU3D(EVAL(AYB(AXB(ndy))));
    z_d[4] = PSU3D(EVAL(AZB(ndz)));

    x_d[5] = PSU3D(EVAL(AYB(ndx)));
    y_d[5] = PSU3D(EVAL(AYB(ndy)));
    z_d[5] = PSU3D(EVAL(AZB(ndz)));

    x_d[6] = PSU3D(EVAL(AXB(ndx)));
    y_d[6] = PSU3D(EVAL(AXB(ndy)));
    z_d[6] = PSU3D(EVAL(AZB(ndz)));

    x_d[7] = PSU3D(dx);
    y_d[7] = PSU3D(dy);
    z_d[7] = PSU3D(EVAL(AZB(ndz)));

    // z_d[7]->display("z_d[7] = ");
    // x_d[1]->display("x_d[0] = ");
    // y_d[1]->display("y_d[0] = ");
    // z_d[1]->display("z_d[0] = ");

    

    // need update ghost of dx, dy, dz
    for (int i = 0; i < 8; i++) {
      vector<MPI_Request> reqs;
      oa::funcs::update_ghost_start(x_d[i], reqs, 3);
      oa::funcs::update_ghost_end(reqs);
      reqs.clear();

      oa::funcs::update_ghost_start(y_d[i], reqs, 3);
      oa::funcs::update_ghost_end(reqs);
      reqs.clear();

      oa::funcs::update_ghost_start(z_d[i], reqs, 3);
      oa::funcs::update_ghost_end(reqs);
      reqs.clear();
    }

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

  // int rk = x_d[1]->rank();
  // Shape sp = x_d[1]->local_shape();
  // sp[0] += 2;
  // sp[1] += 2;
  // sp[2] += 2;

  // MPI_ORDER_START
  // printf("=====%d======\n", rk);
  // oa::utils::print_data(x_d[1]->get_buffer(), sp, DATA_DOUBLE);
  // MPI_ORDER_END

}

ArrayPtr Grid::get_grid_dx(int pos){
  return x_d.at(pos);
}

ArrayPtr Grid::get_grid_dy(int pos){
  return y_d.at(pos);
}

ArrayPtr Grid::get_grid_dz(int pos){
  return z_d.at(pos);
}

ArrayPtr Grid::get_grid(int pos, NodeType t) {
  ArrayPtr np;
  if (pos == -1) return np;
  switch (t) {
  case TYPE_AXB:
  case TYPE_DXB:
  case TYPE_AXF:
  case TYPE_DXF:
  case TYPE_DXC:
    return Grid::global()->get_grid_dx(pos);
    break;
  case TYPE_AYB:
  case TYPE_DYB:
  case TYPE_AYF:
  case TYPE_DYF:
  case TYPE_DYC:
    return Grid::global()->get_grid_dy(pos);
    break;
  case TYPE_AZB:
  case TYPE_DZB:
  case TYPE_AZF:
  case TYPE_DZF:
  case TYPE_DZC:
    return Grid::global()->get_grid_dz(pos);
    break;
  }
  return np;
}

int Grid::get_pos(int pos, NodeType t){
  switch (t) {
  case TYPE_AXB:
  case TYPE_DXB:
  case TYPE_AXF:
  case TYPE_DXF:
  case TYPE_DXC:
    return Grid::global()->get_pos_x(pos);
    break;
  case TYPE_AYB:
  case TYPE_DYB:
  case TYPE_AYF:
  case TYPE_DYF:
  case TYPE_DYC:
    return Grid::global()->get_pos_y(pos);
    break;
  case TYPE_AZB:
  case TYPE_DZB:
  case TYPE_AZF:
  case TYPE_DZF:
  case TYPE_DZC:
    return Grid::global()->get_pos_z(pos);
    break;
  default:
    return pos;
  }
}

int Grid::get_pos_x(int pos) {
  return x_map.at(pos);
}

int Grid::get_pos_y(int pos) {
  return y_map.at(pos);
}

int Grid::get_pos_z(int pos) {
  return z_map.at(pos);
}

Grid* Grid::global() {
  static Grid gd;
  return &gd;
}
