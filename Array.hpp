#include <vector>
#include <memory>
#include "Partition.hpp"

#ifndef ARRAY_HPP
#define ARRAY_HPP

/*
 * Array:
 *      buffer:         data
 *      PartitionPtr:   partition information
 *      BoxPtr:         Array can be a reference one 
 */
class Array;
typedef shared_ptr<Array> ArrayPtr;

class Array {
    private:
        void *m_buffer;
        bool m_is_field = false;
        int m_grid_pos = -1;
        int m_data_type = 2;
        PartitionPtr m_par_ptr;
        BoxPtr m_ref_box_ptr;
        bool m_is_scalar = false;
        bool m_is_seq = false;
        size_t m_hash;

    public:
        Array(PartitionPtr ptr); 
        Array(PartitionPtr ptr, void *data, int data_type = DATA_DOUBLE); 
        int data_type();
        void* buffer();
        void set_buffer(void *buffer, int size);
        PartitionPtr partition();
        void display(const char *prefix = "");
        BoxPtr corners();
        vector<int> local_shape();
        int local_size();
        vector<int> shape();
        int size();
        bool is_scalar();
        void set_hash(size_t hash);
        size_t hash();
};

#endif
