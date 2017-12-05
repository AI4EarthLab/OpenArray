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

    int calc_id(int i, int j, int k, int3 S) {
      int M = S[0];
      int N = S[1];
      int P = S[2];
      return k * M * N + j * M + i;
    }
  }
}
