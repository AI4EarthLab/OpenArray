
#ifndef SUNWAY
#include "../ArgumentParser.hpp"
#include <iostream>

extern "C"{

  void c_oa_option_init(char* cmdline){
    ArgumentParser::global()->set_cmdline(cmdline);
  }

  ///:for ti in ['int', 'double', 'float']
  ///:for tv in ['int', 'double', 'float']  
  void c_oa_option_${ti}$_${tv}$(${ti}$& i, char* key, ${tv}$ v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<${ti}$>(key,
                ${ti}$(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<${ti}$>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  ///:endfor
  ///:endfor
}
#endif
