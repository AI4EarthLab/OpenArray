class Node {
    private:
        int id;
        ArrayPtr data;
        vector<NodePtr> input;
        vector<NodePtr> output;
        ULL hash;
        NodeType type;
        BoxPtr ref;

    public:
        Node(NodePtr u);
        Node(ArrayPtr array);
        ~Node();
        void display(string prefix);
        vector<NodePtr> input();
        vector<NodePtr> output();
        ULL hash();
        int type();

};
