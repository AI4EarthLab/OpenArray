#include "Internal.hpp"

namespace oa {
  namespace internal {
    template<>
    void set_buffer_rand(int *buffer, int size) {
      srand(SEED);
      for (int i = 0; i < size; i++) {
        int r = rand() % 10000;
        buffer[i] = r;
      }
    }
/*
    int calc_id(int i, int j, int k, oa_int3 S) {
      int M = S[0];
      int N = S[1];
      int P = S[2];
      return k * M * N + j * M + i;
    }
*/
    oa_int3 calc_step(Box box, int d, int sw) {
      int xs, xe, ys, ye, zs, ze;
      int st, ed, step;
      box.get_corners(xs, xe, ys, ye, zs, ze);

      switch (d) {
        case 0:
          st = sw;
          ed = xe - xs + sw;
          break;
        case 1:
          st = sw;
          ed = ye - ys + sw;
          break;
        case 2:
          st = sw;
          ed = ze - zs + sw;
          break;
      }
      step = (st + 1 == ed) ? 0 : 1;
      oa_int3 loop;
      loop[0] = st; loop[1] = step; loop[2] = ed;
      return loop;
    }
  }
}
