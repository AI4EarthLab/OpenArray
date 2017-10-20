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

		//sub_A = sub(A, box)
		template <typename T>
		void set_buffer_subarray(T *sub_buffer, T *buffer, const Box &sub_box,
			const Box &box, int sw) {
			
			Shape sp = box.shape(sw);
			int M = sp[0];
			int N = sp[1];
			int P = sp[2];
			
			Box bd_box = box.boundary_box(sw);
			Box ref_box = sub_box.ref_box(bd_box);
			int xs, xe, ys, ye, zs, ze;
			ref_box.get_corners(xs, xe, ys, ye, zs, ze, sw);
			
			//ref_box.display("ref_box");

			int cnt = 0;
			for (int k = zs; k < ze; k++) {
				for (int j = ys; j < ye; j++) {
					for (int i = xs; i < xe; i++) {
						sub_buffer[cnt++] = buffer[k * M * N + j * M + i];
						//cout<<buffer[cnt-1]<<" ";
					}
					//cout<<endl;
				}
				//cout<<endl;
			}
		}

		void set_buffer_rand(int *buffer, int size);

		void set_buffer_seqs(int *buffer, const Shape& s, Box box, int sw); 

		template <typename T>
		void set_ghost_consts(T *buffer, const Shape &sp, T val, int sw = 1) {
			int M = sp[0] + 2 * sw;
			int N = sp[1] + 2 * sw;
			int P = sp[2] + 2 * sw;

			int cnt = 0;
			for (int k = 0; k < P; k++) {
				for (int j = 0; j < N; j++) {
					for (int i = 0; i < M; i++) {
						if ((sw <= k && k < P - sw) &&
								(sw <= j && j < N - sw) &&
								(sw <= i && i < M - sw)) {
							cnt++;
							continue;
						}
						buffer[cnt++] = val;
					}
				}
			}


		}
	}
}

#endif
