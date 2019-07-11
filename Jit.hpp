#ifndef _JIT_HPP__
#define _JIT_HPP__

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#ifndef _WITHOUT_LLVM_
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#endif
#include <memory>
/*
using namespace clang;
using namespace clang::driver;
using namespace std;
using namespace llvm;
*/

class Jit;
typedef std::shared_ptr<Jit> JitPtr;

std::string GetExecutablePath(const char *Argv0); 
class Jit{
  private:
    char * funcname;
    char * code;
    int argc;
    char **argv;
    uint64_t Entry;
#ifndef _WITHOUT_LLVM_
    std::unique_ptr<llvm::ExecutionEngine> JEE;
    llvm::ExecutionEngine * createExecutionEngine(std::unique_ptr<llvm::Module> M, std::string *ErrorStr) ;
    std::unique_ptr<llvm::ExecutionEngine> GetAddress(std::unique_ptr<llvm::Module> Mod); 
#endif

  public:
    Jit(int argcin, char **argvin, char * fnin, char * codein);
    ~Jit();
    uint64_t compile();
};
#endif
