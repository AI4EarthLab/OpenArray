#ifndef __TO_TYPE_HPP__
#define __TO_TYPE_HPP__

#include <iostream>
#include <type_traits>
#include "../common.hpp"

namespace oa {
    namespace utils {

    	//determine data type of T
    	template <typename T>
    	DataType to_type() {
    		if (std::is_same<T, int>::value) return DATA_INT;
    		if (std::is_same<T, float>::value) return DATA_FLOAT;
    		if (std::is_same<T, double>::value) return DATA_DOUBLE;
    	}

    }
}

#endif
