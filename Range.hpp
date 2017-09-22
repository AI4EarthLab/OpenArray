#include<iostream>
#include<string>
using namespace std;

#ifndef RANGE_HPP
#define RANGE_HPP

/*
 * Range [lower, upper)
 * */

class Range {
    private:
        int lower;
        int upper;

    public:
        Range();
        Range(int st, int ed);
        bool equal(int st, int ed);
        bool equal(Range &rg);
        int size();
        void display(char const *prefix = "");

        // check if range is inside the range u
        bool is_inside(Range &u);

        // [lower, upper) + num = [lower + num, upper + num)
        void shift(int num);
};

#endif
