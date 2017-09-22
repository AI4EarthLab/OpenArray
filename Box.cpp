class Box {
    private:
        range rx;
        range ry;
        range rz;
    
    public:
        Box(int[3] starts, int[3] counts);
        Box(range rx, range ry, range rz);
        ~Box();
        int[3] shape();
        int size();
        bool equal(Box& u);
        bool equal_shape(int[3] shape);
        void display(string prefix);
        bool is_inside(Box& u);
        bool intersection(Box& u);
        bool is_valid();
        Box get_intersection_box(Box& u);
        void reset();
         
        

};
