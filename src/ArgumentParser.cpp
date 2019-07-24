#ifndef SUNWAY
#include "ArgumentParser.hpp"
#include "boost/bind.hpp"
#include<boost/algorithm/string/split.hpp>
#include<boost/algorithm/string.hpp>

void ArgumentParser::set_cmdline(char* cmd){
  cmdline = cmd;
}

void ArgumentParser::parse_cmdline(int argc,  char** argv){
  po::store(po::command_line_parser(argc, argv)
          .options(m_ops_desc)
          .allow_unregistered().run(), m_vm);

  po::notify(m_vm);  
}

const po::option_description* ArgumentParser::find(char* key){
  return m_ops_desc.find_nothrow(std::string(key), false);
}


void ArgumentParser::parse_cmdline(){
  using boost::is_any_of;   
  std::vector<std::string> vs;
  boost::algorithm::split(vs, cmdline, is_any_of("\t "));  
  po::store(po::command_line_parser(vs)
          .options(m_ops_desc)
          .allow_unregistered().run(), m_vm);

  po::notify(m_vm);
}

void ArgumentParser::parse_file(char* file){
  
}

void ArgumentParser::show(){
  std::cout<<m_ops_desc;
}
#endif
