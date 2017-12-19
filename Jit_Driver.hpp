#ifndef __JIT_DRIVER_HPP__
#define __JIT_DRIVER_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <functional>
#include "Jit.hpp"

typedef std::shared_ptr<Jit> JitPtr;

using namespace std;

typedef void fusion_kernel_rawptr (void**&, int);
typedef void (*kernel_func)(void**&, int);
typedef std::function<fusion_kernel_rawptr> FusionKernelPtr;

typedef unordered_map<size_t, FusionKernelPtr> FKPtrMap;
typedef unordered_map<size_t, JitPtr> JitPoolMap;

class Jit_Driver{
private:
  FKPtrMap kernel_dict;
  JitPoolMap m_jit_pool;

public:
  FusionKernelPtr get(size_t hash) {
    if (kernel_dict.find(hash) == kernel_dict.end()) return NULL;
    return kernel_dict[hash];
  }

  void insert(size_t hash, const stringstream& code) {
    FusionKernelPtr fk_ptr = get(hash);
    if (fk_ptr != NULL) return ;

    int fack_argc = 4;
    char arg0[] = "-O3" ;
    char arg1[] = "-ffast-math";
    char arg2[] = "-O3";
    char arg3[]= ""; //"-I/home/wangdong/comp/llvm.debug/lib/clang/6.0.0/include";
    //char **fake_argv = new char *[fack_argc+1]{arg0, arg1, arg2, arg3};
    char **fake_argv = new char*[fack_argc];
    for (int i = 0; i < fack_argc; i++) {
        fake_argv[i] = new char[256];
    }
    strcpy(fake_argv[0], arg0);
    strcpy(fake_argv[1], arg1);
    strcpy(fake_argv[2], arg2);
    strcpy(fake_argv[3], arg3);
    
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

    JitPtr jit_ptr = JitPtr(new Jit(fack_argc, fake_argv, cname, ccode));
    m_jit_pool[hash] = jit_ptr;

    uint64_t Entry = jit_ptr->compile();
    
    fk_ptr = (kernel_func)Entry;
    kernel_dict[hash] = fk_ptr;

    delete []ccode;
    delete []cname;
    for (int i = 0; i < fack_argc; i++) delete[] fake_argv[i];
    delete[] fake_argv;

    return ;
  }

  static Jit_Driver* global() {
    static Jit_Driver jit;
    return &jit;
  }
};

#endif
