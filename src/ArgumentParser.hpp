#ifndef SUNWAY
#ifndef __ARGUMENTPARSER_HPP__
#define __ARGUMENTPARSER_HPP__

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <string>
#include <iostream>
#include <boost/tokenizer.hpp>

namespace po = boost::program_options;


class ArgumentParser{
private:
  po::options_description  m_ops_desc;
  po::variables_map m_vm;
  std::string cmdline;
public:
  template<class T>
  T get_option(const char* key){
    return m_vm[key].as<T>();
  }
  
  template<class T>
  void add_option(const char* key,
          T val,
          const char* desc){
    m_ops_desc.add_options()
      (key, po::value<T>()->default_value(val), desc);
  }

  const po::option_description* find(char* key);
  
  void set_cmdline(char* cmd);
  
  void parse_cmdline(int argc, char** argv);
  void parse_cmdline();
    
  void parse_file(char* file);

  /// @brief Tokenize a string.
  /// The tokens will be separated by each non-quoted
  /// space or equal character.
  /// Empty tokens are removed.
  ///
  /// @param input The string to tokenize.
  ///
  /// @return Vector of tokens.
  std::vector<std::string> tokenize(const std::string& input);

  void show();

  static ArgumentParser* global(){
    static ArgumentParser ap;
    return &ap;
  }
};

#endif
#endif
