#include<iostream>
using namespace std;

#ifndef RANGE_HPP
#define RANGE_HPP

/*
 * Range [lower, upper)
 * */

class Range {
    private:
        int m_lower;
        int m_upper;

    public:
        Range();
        Range(int st, int ed);
        bool equal(int st, int ed);
        bool equal(Range &rg);
        int size();
        void display(char const *prefix = "");

        // check if Range is inside the Range u
        bool is_inside(Range &u);

        // [lower, upper) + num = [lower + num, upper + num)
        void shift(int num);

        // check if Range has intersection with Range u
        bool intersection(Range &u); 

        // get intersection Range with Range u
        Range get_intersection(Range &u);
};

#endif
