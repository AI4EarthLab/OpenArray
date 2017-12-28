
#include <memory>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include "../ArgumentParser.hpp"
#include <string>

class Test{
public :
  int i;

  ~Test() {
  std::cout<<"destructor called!"<<std::endl;
  }
};

typedef std::shared_ptr<Test> TestPtr;

int main(int argc, char** argv){

  // TestPtr* ptr = new TestPtr(new Test());

  // // ptr -> i = 1;
  // // std::shared_ptr<Test> ptr1 = ptr;
  // // ptr1 -> i = 2;

  // TestPtr p2 = *ptr;
  // std::cout<< p2->i << std::endl;

  // delete(ptr);
  
  // std::cout<<sizeof(bool) << std::endl;


  ArgumentParser ap;

  ap.add_option<int>("m", 1,"m value");  
  ap.add_option<int>("n", 1,"n value");
  ap.add_option<int>("k", 1,"k value");
  ap.add_option<std::string>("sk", "this is a string", "");
  ap.show();

  ap.parse_cmdline(argc, argv);

  int m = ap.get_option<int>("m");
  int n = ap.get_option<int>("n");
  int k = ap.get_option<int>("k");
  std::string s = ap.get_option<std::string>("sk");
  
  std::cout<<m<<std::endl;  
  std::cout<<n<<std::endl;  
  std::cout<<k<<std::endl;
  std::cout<<s<<std::endl;
}

