
#include "../ArgumentParser.hpp"
#include <iostream>

extern "C"{

  void c_oa_option_init(char* cmdline){
    ArgumentParser::global()->set_cmdline(cmdline);
  }
  
  void c_oa_option_int(int& i, char* key, int v){

    const po::option_description * od =
      ArgumentParser::global()->find(key);

    if(od == NULL){
      ArgumentParser::global()->add_option<int>(key, v, "");
      ArgumentParser::global()->parse_cmdline();
    }

    //ArgumentParser::global()->show();
    
    i = ArgumentParser::global()->get_option<int>(key);
  }
}
