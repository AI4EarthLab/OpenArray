#include<iostream>
#include<vector>
#include "Box.hpp"
using namespace std;

Box :: Box() {}

Box :: Box(Range x, Range y, Range z) :
    rx(x), ry(y), rz(z) {}

Box :: Box(int *starts, int *counts) :
    rx(starts[0], starts[0] + counts[0] - 1),
    ry(starts[1], starts[1] + counts[1] - 1),
    rz(starts[2], starts[2] + counts[2] - 1) {}

Box :: Box(int sx, int ex, int sy, int ey, int sz, int ez) :
    rx(sx, ex), ry(sy, ey), rz(sz, ez) {}

bool Box :: equal(Box &u) {
    return rx.equal(u.rx) && ry.equal(u.ry) && rz.equal(u.rz);
}

bool Box :: equal_shape(Box &b) {
    vector<int> s = shape();
    vector<int> u = b.shape();
    return s[0] == u[0] && s[1] == u[1] && s[2] == u[2]; 
}

void Box :: display(char const *prefix) {
    printf("Box %s: \n", prefix);
    rx.display("rx");
    ry.display("ry");
    rz.display("rz");
}

// check if Box is inside Box u
bool Box :: is_inside(Box &u) {
    return rx.is_inside(u.rx) && ry.is_inside(u.ry) && rz.is_inside(u.rz);
}

// check if Box and Box u has intersection
bool Box :: intersection(Box &u) {
    return rx.intersection(u.rx) && ry.intersection(u.ry) && rz.intersection(u.rz);    
}

// get the intersection box
Box Box :: get_intersection(Box &u) {
    if (!intersection(u)) return Box();
    return Box(rx.get_intersection(u.rx), ry.get_intersection(u.ry), rz.get_intersection(u.rz));
}

// return [shape_x, shape_y, shape_z]
vector<int> Box :: shape() {
    vector<int> s{rx.size(), ry.size(), rz.size()};
    return s; 
}

// return shape_x * shape_y * shape_z
int Box :: size() {
    return rx.size() * ry.size() * rz.size();
}
