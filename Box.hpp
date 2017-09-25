#include<vector>
#include "Range.hpp"

#ifndef BOX_HPP
#define BOX_HPP

/*
 * Box:
 *      dimension: rx, ry, rz
 */

class Box {
    private:
        Range rx, ry, rz;

    public:
        Box();
        Box(Range x, Range y, Range z);
        Box(int* starts, int* counts);
        Box(int sx, int ex, int sy, int ey, int sz, int ez);
        bool equal(Box &u);
        bool equal_shape(Box &b);
        void display(char const *prefix = "");

        // check if Box is inside Box u
        bool is_inside(Box &u);

        // check if Box and Box u has intersection
        bool intersection(Box &u);

        // get the intersection box
        Box get_intersection(Box &u);
        
        // return [shape_x, shape_y, shape_z]
        vector<int> shape();

        // return shape_x * shape_y * shape_z
        int size();
        
};
#endif
