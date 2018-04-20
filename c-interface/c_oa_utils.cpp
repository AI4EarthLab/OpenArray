#include "../Init.hpp"
#include "../common.hpp"
#include "../utils/calcTime.hpp"
#include <mpi.h>
#include "../common.hpp"
#include "../MPI.hpp"

bool g_debug = false;
bool transticbegin = false;

extern "C" {
  // void c_get_rank(int* rank, MPI_Fint fcomm) {
  //     MPI_Comm comm = MPI_Comm_f2c(fcomm);
  //     MPI_Comm_rank(comm, rank);
  // }

  void c_get_rank(int* rank) {
    *rank = MPI_RANK;
  }

  void c_get_size(int* size) {
    *size = MPI_SIZE;
  }
  
  void c_finalize(){
    oa::finalize();
  }

  void c_tic(char* key){
    oa::utils::tic(key);
  }

  void c_toc(char* key){
    oa::utils::toc(key);
  }

  void c_show_timer(){
    oa::utils::show_all();
  }

  void c_open_debug() {
    transticbegin = true;
    g_debug = true;
  }

  void c_close_debug() {
    g_debug = false;
  }
}
