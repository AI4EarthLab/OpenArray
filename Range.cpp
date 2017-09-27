#include<iostream>
#include "Range.hpp"

using namespace std;

Range :: Range() {
    m_lower = m_upper = 0;    
}

Range :: Range(int st, int ed) : 
    m_lower(st), m_upper(ed + 1) {}

bool Range :: equal(int st, int ed) {
    return m_lower == st && m_upper == ed + 1;
}

bool Range :: equal(Range &rg) {
   return m_lower == rg.m_lower && m_upper == rg.m_upper; 
}

int Range :: size() {
    return m_upper - m_lower;
}

void Range :: display(char const *prefix) {
    printf("Range %s is [%d, %d)\n", prefix, m_lower, m_upper);
}

// check if Range is inside the Range u
bool Range :: is_inside(Range &rg) {
    return rg.m_lower <= m_lower && m_upper <= rg.m_upper;
}

// [lower, upper) + num = [lower + num, upper + num)
void Range :: shift(int num) {
    m_lower += num;
    m_upper += num;
}

// check if Range has intersection with Range u
bool Range :: intersection(Range &u) {
	if (m_lower >= u.m_upper || u.m_lower >= m_upper) return false;
	return true;
}

// get intersection Range with Range u
Range Range :: get_intersection(Range &u) {
	if (!intersection(u)) return Range();
	return Range(max(m_lower, u.m_lower), min(m_upper, u.m_upper) - 1);
}