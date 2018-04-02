#include <cstdio>
#include <map>
#include <cstring>
#include <string>
#include <iostream>
#include <limits.h>
#include "gettimeofday.h"

using namespace std;

#define HLINE                                   \
  "----------------------------------------"    \
  "-----------------------------------------"   \
  "-------------------\n"

namespace oa {
  namespace utils {
    map<string, int> cnt;
    map<string, long long> totTime;
    map<string, long long> timer;
    map<string, long long> maxTime;
    map<string, long long> minTime;
    long long tot = 0;
  
//     extern "C"{
// #include "gettimeofday.h"
//       void gettimeofday1(int *ierr2, long long* val);
    
//       void tic(const char* s);
//       void toc(const char* s);
//       void show_time(const char* s);
//       void show_all();
//     }
  
  
    void tic(const char* s) {
      string str(s);
      if (cnt.find(str) == cnt.end()) {
        cnt[str] = 0;
        totTime[str] = 0;
        maxTime[str] = LLONG_MIN;
        minTime[str] = LLONG_MAX;
      }
      cnt[str] += 1;
      int ierr2;
      long long val;
      gettimeofday1(&ierr2, &val);
      timer[str] = val;
    }
  
    void toc(const char* s) {
      string str(s);
      int ierr2;
      long long val;
      gettimeofday1(&ierr2, &val);
      val -= timer[str];
      totTime[str] += val;
      maxTime[str] = max(maxTime[str], val);
      minTime[str] = min(minTime[str], val);
      tot += val;
    }
  
    void show_time(const char* s) {
      string str(s);
      printf("\n");

      printf("%-30s %-15s %-15s %-15s %-15s %-15s\n",
             "function", "min", "max", "call", 
             "total", "average");
      
      printf(HLINE);

      printf("%-30s %-15.6f %-15.6f %-15d %-15.6f %-15.6f\n",
             str.c_str(), minTime[str] / 1e6, 
             maxTime[str] / 1e6, cnt[str],
             totTime[str] / 1e6, totTime[str] / 1e6 / cnt[str]);
      
      printf(HLINE);
      printf("\n");
    }
  
    void show_all() {
      printf("\n");
      printf(HLINE);
      printf("%-30s %-15s %-15s %-15s %-15s %-15s %-15s%%\n",
             "function", "min", "max", "call", 
             "total", "average", "per");

      printf(HLINE);
      
      for (map<string, int>::iterator it = cnt.begin(); it != cnt.end(); it++) {
        string str = it->first;
        printf("%-30s %-15.6f %-15.6f %-15d %-15.6f %-15.6f %-15.6f%%\n",
               str.c_str(), minTime[str] / 1e6, 
               maxTime[str] / 1e6, cnt[str],
               totTime[str] / 1e6, totTime[str] / 1e6 / cnt[str],
               totTime[str] * 100.0 / tot);
      }
      
      printf(HLINE);
      printf("\n");
    }
  }
}

//using namespace oa::utils;
//int main() {
//	tic("A");
//	tic("B");
//	toc("A");
//	toc("B");
//	tic("A");
//	tic("B");
//	toc("A");
//	toc("B");
//	showTime("A");
//	showTime("B");
//  showAll();
//	return 0;
//}

