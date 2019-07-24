
#include "../IO.hpp"
#include "../MPI.hpp"

extern "C"{
  void c_load(ArrayPtr*& A, char* file, char* var){
    if(A == NULL) A = new ArrayPtr();
    //*A = oa::io::load_record(file, var,-1, oa::MPI::global()->comm()); //why? ask yangshaobo!
    *A = oa::io::load(file, var, oa::MPI::global()->comm());
  }

  void c_load_record(ArrayPtr*& A, char* file, char* var, int record){
    if(A == NULL) A = new ArrayPtr();
    record--;
    *A = oa::io::load_record(file, var, record, oa::MPI::global()->comm());
  }

  void c_save(ArrayPtr*& A, char* file, char* var){
    assert(A != NULL);
    oa::io::save(*A, file, var);
  }

  void c_save_record(ArrayPtr*& A, char* file, char* var, int record){
    assert(A != NULL);
    record--;
    oa::io::save_record(*A, file, var,record);
  }
}
