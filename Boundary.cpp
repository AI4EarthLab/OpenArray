class Boundary {
    public:
        void update_ghost(ArrayPtr arrayptr);
        void update_ghost_async(ArrayPtr arrayptr, int* status);
};
