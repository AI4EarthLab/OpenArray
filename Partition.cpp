class Partition {
    private:
        MPI_Comm comm;
        int[3] shape;
        int[3] procs_shape;
        BoundType[3] bound_type;
        StencilType stencil_type;
        int stencil_width;

        vector<int> lx;
        vector<int> ly;
        vector<int> lz;
        vector<int> clx;
        vector<int> cly;
        vector<int> clz;

    public:
        Partition(MPI_Comm comm, int[3] shape, int[3] procs_shape);
        Partition(MPI_Comm comm, vector<int>& lx, vector<int>& ly, vector<int>& lz);
        Partition(Partition &p);
        ~Partition();
        void display(string prefix);
        void disp_distr(string prefix);
        bool equal_distr(Partition &p);
        bool equal(Partition &p);
        Box get_local_box(int[3] coord);
        int[3] shape();
        int size();
        void update_distr(Partition &p, Reference &ref);
        void update_acc_distr();
        
        void set_stencil(int t, int w);
        int local_shape(int[3] coord);
        int get_procs_rank(int x, int y, int z);
        int[3] get_procs_3d(int rank);

};
