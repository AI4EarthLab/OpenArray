
#include "../MPI.hpp"
#include "../Array.hpp"
#include "../Grid.hpp"
#include <iostream>
extern "C"{
  void c_grid_init (char* ch, const ArrayPtr*& A,
          const ArrayPtr*& B, const ArrayPtr*& C){
    Grid::global()->init_grid(*ch, *A, *B, *C);
  }

  void c_grid_bind(ArrayPtr*& A, int pos){
    (*A)->set_pos(pos);
  }
}
