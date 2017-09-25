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
    printf("Range %s is [%d, %d)\n", prefix, lower, upper);
}

// check if Range is inside the Range u
bool Range :: is_inside(Range &rg) {
    return rg.lower <= lower && upper <= rg.upper;
}

// [lower, upper) + num = [lower + num, upper + num)
void Range :: shift(int num) {
    lower += num;
    upper += num;
}

// check if Range has intersection with Range u
bool Range :: intersection(Range &u) {
	if (lower >= u.upper || u.lower >= upper) return false;
	return true;
}

// get intersection Range with Range u
Range Range :: get_intersection(Range &u) {
	if (!intersection(u)) return Range();
	return Range(max(lower, u.lower), min(upper, u.upper) - 1);
}