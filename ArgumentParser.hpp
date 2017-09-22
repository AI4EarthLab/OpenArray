
#ifndef __ARGUMENTPARSER_HPP__
#define __ARGUMENTPARSER_HPP__

#include <boost/program_options.hpp>
#include <boost/format.hpp>

namespace po = boost::program_options;

class ArgumentParser{
private:
  po::options_description  m_ops_desc;
  po::variables_map m_vm;
  
public:
  template<class T>
  void add_option(const char* key, const char* desc = "");

  template<class T>
  T get_option(const char*key);
  
  void parse_cmdline(int argc, char** argv);
  void parse_file(char* file);

  void show();
};

template<class T>
void ArgumentParser::add_option(const char* key, const char* desc){
  m_ops_desc.add_options()
    (key, po::value<T>(), desc);
}

void ArgumentParser::parse_cmdline(int argc,  char** argv){
  po::store(po::parse_command_line(argc, argv, m_ops_desc), m_vm);
  po::notify(m_vm);  
}

void ArgumentParser::parse_file(char* file){
  
}

template<class T>
T ArgumentParser::get_option(const char* key){
  return m_vm[key].as<T>();
}

void ArgumentParser::show(){
  std::cout<<m_ops_desc;
}

#endif
