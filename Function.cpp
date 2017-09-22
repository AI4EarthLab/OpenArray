class Function {
    public:
        Arrayptr consts(int m, int n, int p, int x);
        Arrayptr consts(int m, int n, int p, float x);
        Arrayptr consts(int m, int n, int p, double x);
        ones();
        zeros();
        rand(ArrayPtr &arrayptr, double x);
        seqs(ArrayPtr &arrayptr);
        //transfer(ArrayPtr &A, ArrayPtr &B);
        
};
