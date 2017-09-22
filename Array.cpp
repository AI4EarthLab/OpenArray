class Array {
    private:
        void* buffer;
        bool is_field;
        int grid_pos;
        int data_type;
        PartitionPtr partition;
        GridPtr grid;

    public:
        Array(PartitionPtr partitionptr); 
        Array(ArrayPtr arrayptr);
        ~Array();
        //void update_buffer(void* data);
        int data_type();
        void* buffer();
        PartitionPtr partition();
        void display(string prefix);
        int[3] shape();
        int size();
        int[3] local_shape();


};
