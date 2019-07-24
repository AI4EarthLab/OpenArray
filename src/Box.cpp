/*
 * Box.cpp:
 *
=======================================================*/

#include <vector>
#include <iostream>
#include "Box.hpp"
#include "common.hpp"

using namespace std;

Box::Box() {}

Box::Box(Range x, Range y, Range z) :
  m_rx(x), m_ry(y), m_rz(z) {}

Box::Box(int *starts, int *counts) :
  m_rx(starts[0], starts[0] + counts[0]),
  m_ry(starts[1], starts[1] + counts[1]),
  m_rz(starts[2], starts[2] + counts[2]) {}

Box::Box(int sx, int ex, int sy, int ey, int sz, int ez) :
  m_rx(sx, ex), m_ry(sy, ey), m_rz(sz, ez) {}

int Box::xs() const{
  return m_rx.get_lower();
}

int Box::xe() const{
  return m_rx.get_upper();
}

int Box::ys() const{
  return m_ry.get_lower();
}

int Box::ye() const{
  return m_ry.get_upper();
}

int Box::zs() const{
  return m_rz.get_lower();
}

int Box::ze() const{
  return m_rz.get_upper();
}

Range Box::get_range_x() {
  return m_rx;
}

Range Box::get_range_y() {
  return m_ry;
}

Range Box::get_range_z() {
  return m_rz;
}

void Box::get_corners(
    int &xs, int &xe, int &ys, int &ye, int &zs, int &ze) const{
  xs = m_rx.get_lower();
  xe = m_rx.get_upper();
  ys = m_ry.get_lower();
  ye = m_ry.get_upper();
  zs = m_rz.get_lower();
  ze = m_rz.get_upper();
}

void Box::get_corners_with_stencil(
    int &xs, int &xe, int &ys, int &ye, int &zs, int &ze, int sw) const{
  xs = m_rx.get_lower() - sw;
  xe = m_rx.get_upper() + sw;
  ys = m_ry.get_lower() - sw;
  ye = m_ry.get_upper() + sw;
  zs = m_rz.get_lower() - sw;
  ze = m_rz.get_upper() + sw;
}

bool Box::equal(const Box &u) {
  return m_rx.equal(u.m_rx) && m_ry.equal(u.m_ry) && m_rz.equal(u.m_rz);
}

bool Box::equal_shape(const Box &b) {
  Shape s = shape();
  Shape u = b.shape();
  return s[0] == u[0] && s[1] == u[1] && s[2] == u[2]; 
}

void Box::display(char const *prefix) const {
  printf("Box %s: \n", prefix);
  m_rx.display("rx");
  m_ry.display("ry");
  m_rz.display("rz");
}

bool Box::is_inside(const Box &u) {
  return m_rx.is_inside(u.m_rx) && m_ry.is_inside(u.m_ry) && m_rz.is_inside(u.m_rz);
}

bool Box::intersection(const Box &u) {
  return m_rx.intersection(u.m_rx) && 
  m_ry.intersection(u.m_ry) && m_rz.intersection(u.m_rz);  
}

Box Box::get_intersection(const Box &u){
  if (!intersection(u)) return Box();
  return Box(m_rx.get_intersection(u.m_rx), 
  m_ry.get_intersection(u.m_ry), m_rz.get_intersection(u.m_rz));
}

Box Box::ref_box(const Box &ref) const {
  int _xs, _xe, _ys, _ye, _zs, _ze;
  ref.get_corners(_xs, _xe, _ys, _ye, _zs, _ze);
  Range rx = m_rx, ry = m_ry, rz = m_rz;
  // shift rx, ry, rz based on reference box
  rx.shift(-_xs);
  ry.shift(-_ys);
  rz.shift(-_zs);
  return Box(rx, ry, rz);
}

Box Box::boundary_box(int sw) const {
  int xs, xe, ys, ye, zs, ze;
  get_corners_with_stencil(xs, xe, ys, ye, zs, ze, sw);
  return Box(xs, xe, ys, ye, zs, ze);
}

Shape Box::shape() const {
  Shape s{{m_rx.size(), m_ry.size(), m_rz.size()}};
  return s; 
}

Shape Box::shape_with_stencil(int sw) const {
  Shape s{{m_rx.size() + 2 * sw, m_ry.size() + 2 * sw, m_rz.size() + 2 * sw}};
  return s; 
}

int Box::size() const{
  return m_rx.size() * m_ry.size() * m_rz.size();
}

int Box::size_with_stencil(int sw) const{
  return (m_rx.size() + 2 * sw) * (m_ry.size() + 2 * sw) * 
  (m_rz.size() + 2 * sw);
}

oa_int3 Box::starts() const{
  return {{m_rx.get_lower(),
      m_ry.get_lower(),
      m_rz.get_lower()}};
}

oa_int3 Box::counts() const{
  return {{m_rx.size(), m_ry.size(), m_rz.size()}};
}

Box Box::shift(int i){
  Range rx = m_rx, ry = m_ry, rz = m_rz;
  rx.shift(i);
  ry.shift(i);
  rz.shift(i);
  return Box(rx, ry, rz);
}

Box Box::shift(int x, int y, int z){
  Range rx = m_rx, ry = m_ry, rz = m_rz;
  rx.shift(x);
  ry.shift(y);
  rz.shift(z);
  return Box(rx, ry, rz);
}

