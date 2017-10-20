
#ifndef __NODE_HPP__
#define __NODE_HPP__
#include "Array.hpp"
#include <vector>
#include <memory>

class Node;

typedef std::shared_ptr<Node> NodePtr;
typedef std::vector<NodePtr> NodeList;

class Node {
private:
  int id;
  ArrayPtr data;
  NodeList m_input;
  NodeList m_output;
  size_t m_hash;
  int m_type;
  //BoxPtr ref;

public:
  Node(){};
  Node(NodePtr u);
  //Node(ArrayPtr array);
  ~Node(){};
  void display(std::string prefix){};

  NodePtr input(int i){
    return m_input.at(i);
  };
  
  NodePtr output(int i){
    return m_output.at(i);
  };

  void add_input(int pos, NodePtr in){
    m_input.insert(m_input.begin() + pos, in);
  };

  void add_output(int pos, NodePtr out){
    m_output.push_back(out);
  };

  void set_type(NodeType type){
    m_type = type;
  }

  void set_data(const ArrayPtr& ptr){
    data = ptr;
  }
  
  size_t hash(){};
  int type(){};

  void reset(){
    
  }
};


#endif
