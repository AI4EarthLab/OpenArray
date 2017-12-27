#ifndef __JIT_DRIVER_HPP__
#define __JIT_DRIVER_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <iostream>
#include <sstream>
#include <functional>
#include "Jit.hpp"
#include <vector>
#include <dlfcn.h>  
#include <unistd.h>  

typedef std::shared_ptr<Jit> JitPtr;

using namespace std;

typedef void fusion_kernel_rawptr (void**&, int);
typedef void (*kernel_func)(void**&, int);
typedef std::function<fusion_kernel_rawptr> FusionKernelPtr;

typedef unordered_map<size_t, FusionKernelPtr> FKPtrMap;
typedef unordered_map<size_t, JitPtr> JitPoolMap;

class Jit_Driver{
private:
  int haveicc = -1;
  FKPtrMap kernel_dict;
  JitPoolMap m_jit_pool;

  int insert_icc(size_t hash, const stringstream& code){
    //std::cout<<"icc"<<std::endl;
    FusionKernelPtr fk_ptr = get(hash);
    if (fk_ptr != NULL) return ;
    stringstream filename;
    filename<<"/tmp/"<<"kernel_"<<hash<<".cpp";
    stringstream objname;
    objname<<"/tmp/"<<"kernel_"<<hash<<".so";
    ofstream sourcefile;
    sourcefile.open(filename.str());
    sourcefile<<code.str();
    sourcefile.close();
    stringstream cmd;
    cmd<<"icc -shared -fPIC -nostartfiles -xHost -O3 -Ofast  -g -w -o "<<objname.str().c_str()<<" "<<filename.str().c_str();
    
    if(system(cmd.str().c_str()) != 0)
    {
      std::cout<<"icc compile err"<<std::endl;
      haveicc = 0;
      return -1;
    }
    
    uint64_t Entry ;
    void *dlHandle = NULL;  

    stringstream funcname;
    funcname<<"kernel_"<<hash;
    dlHandle = dlopen(objname.str().c_str(), RTLD_LAZY);  
    while(dlHandle == NULL)  
    {  
      dlHandle = dlopen(objname.str().c_str(), RTLD_LAZY);  
    }  
    char *error = NULL;
    Entry = (uint64_t)dlsym(dlHandle, funcname.str().c_str());  
    if((error = dlerror()) != NULL)  
    {  
      printf("dlsym error(%s).\n", error);  
      return -1;  
    }  


    fk_ptr = (kernel_func)Entry;
    kernel_dict[hash] = fk_ptr;

    return 1;

  }
public:
  FusionKernelPtr get(size_t hash) {
    if (kernel_dict.find(hash) == kernel_dict.end()) return NULL;
    return kernel_dict[hash];
  }

  void testicc(){
    if(system("icc -v 2> /tmp/haveicc.log") == 0)
    {
      haveicc = 1;
      std::cout<<"use icc."<<std::endl;
    }
    else
    {
      haveicc = 0;
      std::cout<<"use llvm."<<std::endl;
    }
  }

  void insert(size_t hash, const stringstream& code) {
    if(haveicc == -1) 
      testicc();
    if(haveicc){
      if(insert_icc(hash,code) == 1)
        return;
    }
    //std::cout<<"llvm"<<std::endl;
    FusionKernelPtr fk_ptr = get(hash);
    if (fk_ptr != NULL) return ;

    std::vector<string> opvs;

    opvs.push_back("-O3");
    opvs.push_back("-Ofast");
    opvs.push_back("-ffast-math");
    opvs.push_back("-march=core-avx2");
    opvs.push_back("-m64");
    opvs.push_back("--std=c++0x");

    char **fake_argv = new char*[opvs.size()];

    for (int i = 0; i < opvs.size(); i++) {
        fake_argv[i] = new char[256];
        strcpy(fake_argv[i], opvs[i].c_str());
    }
    
    stringstream ss;
    ss<<"kernel_"<<hash;
    
    const string& scode = code.str();  
    const char* cccode = scode.c_str(); 
    const string& sname = ss.str();  
    const char* ccname = sname.c_str(); 
    
    char *ccode = new char[strlen(cccode)+1];
    char *cname = new char[strlen(ccname)+1];
    strcpy(ccode, cccode);
    strcpy(cname, ccname);

    JitPtr jit_ptr = JitPtr(new Jit(opvs.size(), fake_argv, cname, ccode));
    m_jit_pool[hash] = jit_ptr;

    uint64_t Entry = jit_ptr->compile();
    
    fk_ptr = (kernel_func)Entry;
    kernel_dict[hash] = fk_ptr;

    delete []ccode;
    delete []cname;
    for (int i = 0; i < opvs.size(); i++) delete[] fake_argv[i];
    delete[] fake_argv;

    return ;
  }

  static Jit_Driver* global() {
    static Jit_Driver jit;
    return &jit;
  }
};

#endif
