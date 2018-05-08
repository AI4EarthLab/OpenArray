
#include "../Array.hpp"
#include "../Init.hpp"
#ifndef SUNWAY
#include "../ArgumentParser.hpp"
#endif
#include "../log.hpp"

extern "C"{
  void c_init(int fcomm, int* procs_shape, char* cmdline){
    Shape ps;
    ps[0] = procs_shape[0];
    ps[1] = procs_shape[1];
    ps[2] = procs_shape[2];
    oa::init(fcomm, ps);

#ifndef SUNWAY
    ArgumentParser::global()->set_cmdline(cmdline);
#endif
    //oa::logging::logging_start(1);
  }
}
