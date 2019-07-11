
#include "../Partition.hpp"

extern "C"{
  void c_get_default_procs_shape(int* f){
    //printf("%p\n", f);
    Shape s = Partition::get_default_procs_shape();
    f[0] = s[0];
    f[1] = s[1];
    f[2] = s[2];
  };
  
  void c_set_default_procs_shape(int* s){
    Partition::set_default_procs_shape({{s[0], s[1], s[2]}});
  };
  
  void c_set_auto_procs_shape(){
    Partition::set_default_procs_shape({{0, 0, 0}});
  };
}
