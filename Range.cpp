class Range {
    private:
        int lower;
        int upper;

    public:
        range();
        range(int st, int ed);
        bool equal(int[2] rg);
        bool equal(range rg);
        void shift(int num);
        int size();
        bool is_inside(range &u);
};
