#ifndef __INTERNAL_HPP__
#define __INTERNAL_HPP__

namespace oa {
	namespace internal {
		template <typename T>
		void set_buffer_consts(T *buffer, int size, T val) {
			for (int i = 0; i < size; i++) buffer[i] = val;
		}
	}
}

#endif