#ifndef __JIT_DRIVER_HPP__
#define __JIT_DRIVER_HPP__
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/limits.h>  
#include <errno.h>
#include <sys/wait.h>
#include <iostream>
#include <sstream>
#include <functional>
#include "Jit.hpp"
#include <vector>
#include <dlfcn.h>  
#include <unistd.h>  
#include <sys/stat.h>

typedef std::shared_ptr<Jit> JitPtr;

using namespace std;

typedef void fusion_kernel_rawptr (void**&, int);
typedef void (*kernel_func)(void**&, int);
typedef std::function<fusion_kernel_rawptr> FusionKernelPtr;

typedef unordered_map<size_t, FusionKernelPtr> FKPtrMap;
typedef unordered_map<size_t, JitPtr> JitPoolMap;

class Jit_Driver{
  private:
    int havegcc = -1;
    FKPtrMap kernel_dict;
    JitPoolMap m_jit_pool;

    int insert_gcc(size_t hash, const stringstream& code){
      int myrank;
      //std::cout<<"gcc"<<std::endl;
      FusionKernelPtr fk_ptr = get(hash);
      if (fk_ptr != NULL) return -1;
      stringstream pathname;
      char current_absolute_path[PATH_MAX];
      int cnt = readlink("/proc/self/exe", current_absolute_path, PATH_MAX);
      if (cnt < 0 || cnt >= PATH_MAX)
      {
        printf("***Error***\n");
      }
      current_absolute_path[cnt]='\0';
      int i;
      for (i = cnt-1; i >=0; --i)
      {
        if (current_absolute_path[i] == '/')
        {
          current_absolute_path[i+1] = '\0';
          break;
        }
      }
      pathname<<current_absolute_path;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      pathname<<"kernel_folder";//<<myrank;
      if(myrank == 0)
        if(access(pathname.str().c_str(),0)==-1)
        {
          if (mkdir(pathname.str().c_str(),0777))
          {
            printf("creat folder failed!!!");
          }
        }

      stringstream filename;
      filename<<pathname.str()<<"/kernel_"<<hash<<".cpp";
      stringstream objname;
      objname<<pathname.str()<<"/kernel_"<<hash<<".so";
      ofstream sourcefile;
      stringstream cmd;
      //cmd<<"gcc -shared -fPIC -nostartfiles -O0 -finline -inline-level=2 -finline-functions -no-inline-factor -g -w -o "<<objname.str().c_str()<<" "<<filename.str().c_str();
      cmd<<"gcc -shared -fPIC -nostartfiles -O3 -g -w -o "<<objname.str().c_str()<<" "<<filename.str().c_str();
      //cmd<<"pwd";

      //cout<<objname.str().c_str()<<endl;

      if(myrank == 0)
      {

        bool needcompile = true;
        if(access(filename.str().c_str(), F_OK) == 0)
        {
          ifstream oldfile;
          oldfile.open(filename.str());
          stringstream oldbuffer;
          oldbuffer << oldfile.rdbuf();
          if(oldbuffer.str() == code.str() && access(objname.str().c_str(), F_OK) == 0)
          {
            needcompile = false;
            //cout<<"do not need to compile"<<endl;
          }
        }

        if(needcompile)
        {
          //if(access(filename.str().c_str(), F_OK) == -1)
          {  
            sourcefile.open(filename.str());
            sourcefile<<code.str();
            sourcefile.close();
          }   
          //if(access(objname.str().c_str(), F_OK) == -1)
          {  
            if(system(cmd.str().c_str()) != 0)
            {
              std::cout<<"gcc compile err"<<std::endl;
              havegcc = 0;
              return -1;
            }
          }   
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      uint64_t Entry ;
      void *dlHandle = NULL;  

      stringstream funcname;
      funcname<<"kernel_"<<hash;
      dlHandle = dlopen(objname.str().c_str(), RTLD_LAZY);  
      int dltime = 0;
      while(dlHandle == NULL)  
      {  
        if(system(cmd.str().c_str()) != 0)
        {
          std::cout<<"gcc compile err"<<std::endl;
          havegcc = 0;
          return -1;
        }

        dltime++;
        dlHandle = dlopen(objname.str().c_str(), RTLD_LAZY);  
        std::cout<<"dlopen again "<<dltime<<std::endl;
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

    void testgcc(){
      if(system("gcc -v 2> /dev/null") == 0)
      {
        havegcc = 1;
        // std::cout<<"use gcc."<<std::endl;
      }
      else
      {
        havegcc = 0;
        //std::cout<<"use llvm."<<std::endl;
      }
    }

    void insert(size_t hash, const stringstream& code) {
      if(havegcc == -1) 
        testgcc();
      if(havegcc){
        if(insert_gcc(hash,code) == 1)
          return;
      }
#ifndef _WITHOUT_LLVM_
      //std::cout<<"llvm"<<std::endl;
      stringstream filename;
      filename<<"/home/siofive/GOMO/tmp/"<<"kernel_"<<hash<<".cpp";
      ofstream sourcefile;
      sourcefile.open(filename.str());
      sourcefile<<code.str();
      sourcefile.close();
      FusionKernelPtr fk_ptr = get(hash);
      if (fk_ptr != NULL) return ;

      std::vector<string> opvs;

      opvs.push_back("-O3");
      opvs.push_back("-Ofast");
      opvs.push_back("-ffast-math");
      opvs.push_back("-march=core-avx2");
      opvs.push_back("-m64");
      opvs.push_back("--std=c++0x");
      //opvs.push_back("-Rpass-missed=loop-vectorize");
      //opvs.push_back("-Rpass=loop-vectorize");
      //opvs.push_back("-Rpass=inline");
      //opvs.push_back("-Rpass-missed=inline");
      //opvs.push_back("-m64");

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
#endif
      return ;
    }

    static Jit_Driver* global() {
      static Jit_Driver jit;
      return &jit;
    }
};

#endif
