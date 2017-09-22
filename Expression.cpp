class Expression {
    private:

    public:
        void gen_kernels(NodePtr nodeptr);
        void gen_hash(NodePtr nodeptr);
        void eval(Arrayptr result, NodePtr nodeptr);
        void write_graph(string filename);

};
