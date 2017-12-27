
#include "../IO.hpp"
#include "../MPI.hpp"

extern "C"{
  void c_load(ArrayPtr*& A, char* file, char* var){
    if(A == NULL) A = new ArrayPtr();
    *A = oa::io::load(file, var, oa::MPI::global()->comm());
  }

  void c_save(ArrayPtr*& A, char* file, char* var){
    assert(A != NULL);
    oa::io::save(*A, file, var);
  }
}
