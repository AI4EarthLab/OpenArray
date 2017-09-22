#include<iostream>
#include "Range.hpp"

using namespace std;

Range :: Range() {
    lower = upper = 0;    
}

Range :: Range(int st, int ed) : 
    lower(st), upper(ed + 1) {}

bool Range :: equal(int st, int ed) {
    return lower == st && upper == ed + 1;
}

bool Range :: equal(Range &rg) {
   return lower == rg.lower && upper == rg.upper; 
}

int Range :: size() {
    return upper - lower;
}

void Range :: display(char const *prefix) {
    printf("%s Range is [%d, %d)\n", prefix, lower, upper);
}

bool Range :: is_inside(Range &rg) {
    return rg.lower <= lower && upper <= rg.upper;
}

void Range :: shift(int num) {
    lower += num;
    upper += num;
}
