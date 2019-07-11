
#ifndef SUNWAY
#include "../ArgumentParser.hpp"
#include <iostream>

extern "C"{

  void c_oa_option_init(char* cmdline){
    ArgumentParser::global()->set_cmdline(cmdline);
  }

  void c_oa_option_int_int(int& i, char* key, int v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<int>(key,
                int(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<int>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_int_double(int& i, char* key, double v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<int>(key,
                int(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<int>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_int_float(int& i, char* key, float v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<int>(key,
                int(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<int>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_double_int(double& i, char* key, int v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<double>(key,
                double(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<double>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_double_double(double& i, char* key, double v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<double>(key,
                double(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<double>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_double_float(double& i, char* key, float v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<double>(key,
                double(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<double>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_float_int(float& i, char* key, int v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<float>(key,
                float(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<float>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_float_double(float& i, char* key, double v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<float>(key,
                float(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<float>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
  void c_oa_option_float_float(float& i, char* key, float v){
    try{
      const po::option_description * od =
        ArgumentParser::global()->find(key);

      if(od == NULL){
        ArgumentParser::global()->add_option<float>(key,
                float(v), "");
      
        ArgumentParser::global()->parse_cmdline();
      }

      //ArgumentParser::global()->show();
    
      i = ArgumentParser::global()->get_option<float>(key);
    }catch(const std::exception& e){
      std::cout<<"Exception occured while trying to get option: "
               <<e.what()<<std::endl;
    }
  }
}
#endif
