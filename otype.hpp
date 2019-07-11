#ifndef __OTYPE_HPP__
#define __OTYPE_HPP__

template<class T1, class T2>
struct otype{
  typedef T1 value;
};

template<>
struct otype<int, int>{
  typedef int value;
};

template<>
struct otype<int, float>{
  typedef float value;
};

template<>
struct otype<float, int>{
  typedef float value;
};

template<>
struct otype<int, double>{
  typedef double value;
};

template<>
struct otype<double, int>{
  typedef double value;
};

template<>
struct otype<float, double>{
  typedef double value;
};

template<>
struct otype<double, float>{
  typedef double value;
};

template<>
struct otype<double, double>{
  typedef double value;
};

template<>
struct otype<float, float>{
  typedef float value;
};

#endif
