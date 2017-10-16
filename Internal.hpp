#ifndef __INTERNAL_HPP__
#define __INTERNAL_HPP__

#include <random>
#include "common.hpp"
#include "Box.hpp"

namespace oa {
	namespace internal {
		template <typename T>
		void set_buffer_consts(T *buffer, int size, T val) {
			for (int i = 0; i < size; i++) buffer[i] = val;
		}

		void set_buffer_rand(int *buffer, int size) {
			srand(SEED);
			for (int i = 0; i < size; i++) buffer[i] = rand();
		}

		void set_buffer_seqs(int *buffer, BoxPtr bp, int sw) {
			int cnt = 0;
			int xs, xe, ys, ye, zs, ze;
			bp->get_corners(xs, xe, ys, ye, zs, ze, sw);
			for (int i = xs; i < xe; i++) {
				for (int j = ys; j < ye; j++) {
					for (int k = zs; k < ze; k++) {
						
					}
				}
			}
		}
	}
}

#endif