#include <iostream>
#include <vector>
#include "common.hpp"
#include "Box.hpp"

using namespace std;

Box::Box() {}

Box::Box(Range x, Range y, Range z) :
    m_rx(x), m_ry(y), m_rz(z) {}

Box::Box(int *starts, int *counts) :
    m_rx(starts[0], starts[0] + counts[0] - 1),
    m_ry(starts[1], starts[1] + counts[1] - 1),
    m_rz(starts[2], starts[2] + counts[2] - 1) {}

Box::Box(int sx, int ex, int sy, int ey, int sz, int ez) :
    m_rx(sx, ex), m_ry(sy, ey), m_rz(sz, ez) {}

bool Box::equal(const Box &u) {
    return m_rx.equal(u.m_rx) && m_ry.equal(u.m_ry) && m_rz.equal(u.m_rz);
}

bool Box::equal_shape(const Box &b) {
    Shape s = shape();
    Shape u = b.shape();
    return s[0] == u[0] && s[1] == u[1] && s[2] == u[2]; 
}

void Box::display(char const *prefix) {
    printf("Box %s: \n", prefix);
    m_rx.display("rx");
    m_ry.display("ry");
    m_rz.display("rz");
}

// check if Box is inside Box u
bool Box::is_inside(const Box &u) {
    return m_rx.is_inside(u.m_rx) && m_ry.is_inside(u.m_ry) && m_rz.is_inside(u.m_rz);
}

// check if Box and Box u has intersection
bool Box::intersection(const Box &u) {
    return m_rx.intersection(u.m_rx) && 
    m_ry.intersection(u.m_ry) && m_rz.intersection(u.m_rz);    
}

// get the intersection box
Box Box::get_intersection(const Box &u) {
    if (!intersection(u)) return Box();
    return Box(m_rx.get_intersection(u.m_rx), 
        m_ry.get_intersection(u.m_ry), m_rz.get_intersection(u.m_rz));
}

// return [shape_x, shape_y, shape_z]
Shape Box::shape() const {
    Shape s{m_rx.size(), m_ry.size(), m_rz.size()};
    return s; 
}

// return shape_x * shape_y * shape_z
int Box::size(int sw) {
    return (m_rx.size() + 2 * sw) * (m_ry.size() + 2 * sw) * 
        (m_rz.size() + 2 * sw);
}
