#include "Internal.hpp"

namespace oa {
	namespace internal {
		
		void set_buffer_rand(int *buffer, int size) {
			srand(SEED);
			for (int i = 0; i < size; i++) buffer[i] = rand();
		}

		void set_buffer_seqs(int *buffer, const Shape& s, Box box, int sw) {
			int cnt = 0;
			int xs, xe, ys, ye, zs, ze;
			int M = s[0];
			int N = s[1];
			int P = s[2];
			//cout<<M<<" "<<N<<" "<<P<<endl;
			box.get_corners(xs, xe, ys, ye, zs, ze, sw);
			//printf("%d %d %d %d %d %d\n", xs, xe, ys, ye, zs, ze);
			for (int k = zs; k < ze; k++) {
				for (int j = ys; j < ye; j++) {
					for (int i = xs; i < xe; i++) {
						buffer[cnt++] = k * M * N + j * M + i;
						//cout<<buffer[cnt-1]<<" ";
					}
					//cout<<endl;
				}
				//cout<<endl;
			}
		}

	}
}