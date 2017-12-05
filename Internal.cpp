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
  }
}
