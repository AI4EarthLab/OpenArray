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
      // if(comm == MPI_COMM_SELF) ap->set_seqs();
      // if(ap->shape() == SCALAR_SHAPE) ap->set_scalar();
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
      // if(comm == MPI_COMM_SELF) ap->set_seqs();
      // if(ap->shape() == SCALAR_SHAPE) ap->set_scalar();
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

    inline int calc_id(int i, int j, int k, int3 S);

    void calc_inside(ArrayPtr &ap, ArrayPtr &A, int3 lbound, int3 rbound);

    void calc_outside(ArrayPtr &ap, ArrayPtr &B, int3 lbound, int3 rbound);

    // convert a mpi array to sequential array
    ArrayPtr to_rank0(ArrayPtr A);

    template<class T>
    bool is_equal(const ArrayPtr& A, const arma::Cube<T>& B){

      assert(A->get_data_type() == oa::utils::dtype<T>::type);

      //std::cout<<"is_seqs : " << A->is_seqs() << std::endl;
      
      if (!A->is_seqs()) return false;

      int A_size = A->size();

      // std::cout<<"A_size : "<<A_size;
      // std::cout<<"B_size : "<<arma::size(B)[0]
      //  * arma::size(B)[1] * arma::size(B)[2];

      if (arma::size(B)[0]
          * arma::size(B)[1]
          * arma::size(B)[2] != A_size) {
            return false;
      }

      T* A_buf = (T*)A->get_buffer();
      T* B_buf = (T*)B.memptr();

      for(int i = 0; i < A_size; ++ i) {
        if(abs(A_buf[i] - B_buf[i]) > 1E-6) {
          std::cout<<A_buf[i]<<std::endl;
          std::cout<<B_buf[i]<<std::endl;   
          return false;
        }
      }
      return true;
    }

    template<class T>
    bool is_equal(const ArrayPtr& A, T B){
      
      if (!A->is_seqs_scalar()) return false;
      
      T* A_buf = (T*)A->get_buffer();

      return *A_buf == B;
    }

    template<class T>
    bool is_equal(const ArrayPtr& A, T* B){

      if (!A->is_seqs()) return false;

      ///:for t in [['DATA_INT', 'int'],['DATA_FLOAT','float'],['DATA_DOUBLE','double']]
      if(A->get_data_type() == ${t[0]}$){
        ${t[1]}$* A_buf = (${t[1]}$*)A->get_buffer();
        Shape s = A->buffer_shape();
        const int sw = A->get_partition()->get_stencil_width();

        int cnt = 0;
        for(int k = sw; k < s[2] - sw; k++){
          for(int j = sw; j < s[1] - sw; j++){
            for(int i = sw; i < s[0] - sw; i++){
              if(abs(int(A_buf[i + j * s[0] + k * s[0] * s[1]]) - int(B[cnt])) > 1E-6){
                std::cout<<"compare: "
                         <<int(A_buf[i + j * s[0] + k * s[0] * s[1]])
                         <<"  "
                         <<int(B[cnt])
                         <<std::endl;
                return false;
              }
              cnt ++;
            }
          }
        }
      }
      ///:endfor
      return true;
    }
    
    bool is_equal(const ArrayPtr& A, const ArrayPtr& B);

    template<class T>
    ArrayPtr get_seq_scalar(T val) {
      return consts<T>(MPI_COMM_SELF,SCALAR_SHAPE, val, 0);
    }

    template<class T>
    ArrayPtr get_seq_array(T* val, const Shape& s){
      ArrayPtr a = consts<T>(MPI_COMM_SELF, s, 0, 0);
      const int size = s[0] * s[1] * s[2];
      oa::internal::copy_buffer((T*)a->get_buffer(), val, size);
      return a;
    }

    // set sub(A) = B
    void set(ArrayPtr& A, const Box& A_box, const ArrayPtr& B);

    // set sub(A) = sub(B)
    void set(ArrayPtr& A, const Box& box_a, 
        const ArrayPtr& B, const Box& box_b);

    // set sub(A) = const
    template<typename T>
    void set_with_const(ArrayPtr& A, const Box& A_box, T val) {
      // sub(A)'s partition
      vector<int> rsx, rsy, rsz;
      PartitionPtr pp = A->get_partition();
      Shape ps = pp->procs_shape();
      pp->split_box_procs(A_box, rsx, rsy, rsz);
      
      vector<int> x(ps[0], 0), y(ps[1], 0), z(ps[2], 0);
      for (int i = 0; i < rsx.size(); i += 3)
        x[rsx[i + 2]] = rsx[i + 1] - rsx[i];
      for (int i = 0; i < rsy.size(); i += 3)
        y[rsy[i + 2]] = rsy[i + 1] - rsy[i];
      for (int i = 0; i < rsz.size(); i += 3)
        z[rsz[i + 2]] = rsz[i + 1] - rsz[i];

      int rk = pp->rank();
      vector<int> procs_coord = pp->get_procs_3d(rk);

      int idx = procs_coord[0] - rsx[2];
      int idy = procs_coord[1] - rsy[2];
      int idz = procs_coord[2] - rsz[2];

      // check whether there is local data in process
      if (x[procs_coord[0]] * y[procs_coord[1]] * z[procs_coord[2]] == 0) return ;
      
      Box box = A->get_local_box();
      Box sub_box(
                  rsx[idx * 3], rsx[idx * 3 + 1] - 1,
                  rsy[idy * 3], rsy[idy * 3 + 1] - 1, 
                  rsz[idz * 3], rsz[idz * 3 + 1] - 1
                  );

      // different data_type

      ///:set TYPE = [['DATA_INT', 'int'], ['DATA_FLOAT', 'float'], ['DATA_DOUBLE', 'double']]
      ///:for i in TYPE
      if (A->get_data_type() == ${i[0]}$) {
        oa::internal::set_buffer_subarray_const<${i[1]}$, T>(
          (${i[1]}$*) A->get_buffer(),
          val,
          box,
          sub_box,
          pp->get_stencil_width()
        );
      }

      ///:endfor
      
    }

    ArrayPtr l2g(ArrayPtr& lap);
    ArrayPtr g2l(ArrayPtr& gap);

    // sub(A) = B (MPI_COMM_SELF)
    void set_l2g(ArrayPtr& A, const Box& A_box, ArrayPtr& B);

    // local_A (MPI_COMM_SELF)= sub(global_B)
    void set_g2l(ArrayPtr& local, const Box& sub_box, ArrayPtr& global); 

    // rep arry_B = rep_A(arryA, 1, 2, 3)
    ArrayPtr rep(ArrayPtr& A, int x, int y, int z);
  }
}

#endif
