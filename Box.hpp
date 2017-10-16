#ifndef __BOX_HPP__
#define __BOX_HPP__

#include <vector>
#include <memory>
#include "common.hpp"
#include "Range.hpp"

/*
 * Box:
 *      dimension: rx, ry, rz
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
        void get_corners(int &xs, int &xe, int &ys, int &ye, int &zs, int &ze, int sw = 0);

        bool equal(const Box &u);
        bool equal_shape(const Box &b);
        void display(char const *prefix = "");

        // check if Box is inside Box u
        bool is_inside(const Box &u);

        // check if Box and Box u has intersection
        bool intersection(const Box &u);

        // get the intersection box
        Box get_intersection(const Box &u);
        
        // return [shape_x, shape_y, shape_z]
        Shape shape() const;

        // return shape_x * shape_y * shape_z
        int size(int stencil_width = 0);
        
};

typedef std::shared_ptr<Box> BoxPtr;

#endif
