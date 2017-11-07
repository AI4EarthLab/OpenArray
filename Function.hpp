#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "utils/utils.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"

namespace oa {
  namespace funcs {

    //create an array with const val
    template <typename T>
    ArrayPtr consts(MPI_Comm comm, const Shape& s, T val, int stencil_width = 1) {
      int data_type = oa::utils::to_type<T>();
      ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size(stencil_width);
      oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
      if(comm == MPI_COMM_SELF) ap->set_seqs();
      if(ap->shape() == SCALAR_SHAPE) ap->set_scalar();
      return ap;
    }

    //create an array with const val
    template <typename T>
    ArrayPtr consts(MPI_Comm comm, const vector<int>& x, const vector<int>& y, 
      const vector<int>&z, T val, int stencil_width = 1) {
      int data_type = oa::utils::to_type<T>();
      ArrayPtr ap = ArrayPool::global()->get(comm, x, y, z, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size(stencil_width);
      oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
      if(comm == MPI_COMM_SELF) ap->set_seqs();
      if(ap->shape() == SCALAR_SHAPE) ap->set_scalar();
      return ap;
    }

    // create a ones array
    ArrayPtr ones(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a zeros array
    ArrayPtr zeros(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a rand array
    ArrayPtr rand(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a seqs array
    ArrayPtr seqs(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT); 

    ArrayPtr seqs(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
      const vector<int> &z, int stencil_width = 1, int data_type = DATA_INT);

    // according to partiton pp, transfer src to A
    // A = transfer(src, pp)
    ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp);

    // get a sub Array based on Box b
    ArrayPtr subarray(const ArrayPtr &ap, const Box &box);

    /*
     * update boundary, direction = -1, all dimension
     * direction = 0, dimension x
     * direction = 1, dimension y
     * direction = 2, dimension z
     */
    void update_ghost_start(ArrayPtr ap, vector<MPI_Request> &reqs, int direction = -1);
		
    void update_ghost_end(vector<MPI_Request> &reqs);

    // convert a mpi array to sequential array
    ArrayPtr to_rank0(ArrayPtr A);

    template<class T>
    bool is_equal(const ArrayPtr& A, const arma::Cube<T>& B){

      assert(A->get_data_type() == oa::utils::dtype<T>::type);

      //std::cout<<"is_seqs : " << A->is_seqs() << std::endl;
      
      if(!A->is_seqs())
	return false;

      int A_size = A->size();

      // std::cout<<"A_size : "<<A_size;

      // std::cout<<"B_size : "<<arma::size(B)[0]
      // 	* arma::size(B)[1] * arma::size(B)[2];

      if(arma::size(B)[0]
	 * arma::size(B)[1]
	 * arma::size(B)[2] != A_size){
	return false;
      }

      T* A_buf = (T*)A->get_buffer();
      T* B_buf = (T*)B.memptr();
      for(int i = 0; i < A_size; ++ i){
	if(abs(A_buf[i] - B_buf[i]) > 1E-8){
	  std::cout<<A_buf[i]<<std::endl;
	  std::cout<<B_buf[i]<<std::endl;	  
	  return false;
	}
      }
      return true;
    }

  }
}

#endif
