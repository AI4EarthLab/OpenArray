/*
 * Function.hpp
 * some basic functions in OpenArray
 *
=======================================================*/

#ifndef __FUNCTION_HPP__
#define __FUNCTION_HPP__

#include "common.hpp"
#include "Internal.hpp"
#include "ArrayPool.hpp"
#include "utils/utils.hpp"

namespace oa {
  namespace funcs {

    //create an array with const val
    template <typename T>
    ArrayPtr consts(MPI_Comm comm, const Shape& s, T val, int stencil_width = 1) {
      int data_type = oa::utils::to_type<T>();
      ArrayPtr ap = ArrayPool::global()->get(comm, s, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size_with_stencil(stencil_width);
      oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
      return ap;
    }

    //create an array with const val
    template <typename T>
    ArrayPtr consts(MPI_Comm comm, const vector<int>& x, const vector<int>& y, 
      const vector<int>&z, T val, int stencil_width = 1) {
      int data_type = oa::utils::to_type<T>();
      ArrayPtr ap = ArrayPool::global()->get(comm, x, y, z, stencil_width, data_type);
      Box box = ap->get_local_box();
      int size = box.size_with_stencil(stencil_width);
      oa::internal::set_buffer_consts((T*)ap->get_buffer(), size, val);
      return ap;
    }

    // create a ones array
    ArrayPtr ones(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a zeros array
    ArrayPtr zeros(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a rand array
    ArrayPtr rands(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT);

    // create a seqs array
    ArrayPtr seqs(MPI_Comm comm, const Shape& s, 
      int stencil_width = 1, int data_type = DATA_INT); 

    ArrayPtr seqs(MPI_Comm comm, const vector<int> &x, const vector<int> &y, 
      const vector<int> &z, int stencil_width = 1, int data_type = DATA_INT);

    // according to partiton pp, transfer src to A
    // A = transfer(src, pp)
    ArrayPtr transfer(const ArrayPtr &src, const PartitionPtr &pp);

    template <typename T>
    void local_sub(const ArrayPtr &ap, int x, int y, int z, T* val)  {
      Box b(x, x+1, y, y+1, z, z+1);
      Box local_box = ap->get_local_box();
      int sw = ap->get_stencil_width();
      x -= local_box.xs();
      y -= local_box.ys();
      z -= local_box.zs();
      switch(ap->get_data_type()) {
      case DATA_INT:
        *val = oa::internal::get_buffer_local_sub((int*)ap->get_buffer(), 
            local_box, x, y, z, sw);
        break;
      case DATA_FLOAT:
        *val = oa::internal::get_buffer_local_sub((float*)ap->get_buffer(), 
            local_box, x, y, z, sw);
        break;
      case DATA_DOUBLE:
        *val = oa::internal::get_buffer_local_sub((double*)ap->get_buffer(), 
            local_box, x, y, z, sw);
        break;
      }
    }

    template <typename T>
    void set_local(const ArrayPtr &ap, int x, int y, int z, T val) {
      Box b(x, x+1, y, y+1, z, z+1);
      Box local_box = ap->get_local_box();
      int sw = ap->get_stencil_width();
      x -= local_box.xs();
      y -= local_box.ys();
      z -= local_box.zs();
      switch(ap->get_data_type()) {
      case DATA_INT:
        oa::internal::set_buffer_local((int*)ap->get_buffer(), 
            local_box, x, y, z, (int)val, sw);
        break;
      case DATA_FLOAT:
        oa::internal::set_buffer_local((float*)ap->get_buffer(), 
            local_box, x, y, z, (float)val, sw);
        break;
      case DATA_DOUBLE:
        oa::internal::set_buffer_local((double*)ap->get_buffer(), 
            local_box, x, y, z, (double)val, sw);
        break;
      }
    }

    // get a sub Array based on Box b
    ArrayPtr subarray(const ArrayPtr &ap, const Box &box);

    void update_ghost(ArrayPtr ap);

    /*
     * update boundary, direction = -1, all dimension
     * direction = 0, dimension x
     * direction = 1, dimension y
     * direction = 2, dimension z
     */
    void update_ghost_start(ArrayPtr ap, vector<MPI_Request> &reqs, int direction = -1, 
        int3 lb = {{0,0,0}}, int3 rb = {{0,0,0}}); 
    
    void update_ghost_end(vector<MPI_Request> &reqs);

    void set_ghost_zeros(ArrayPtr ap);

    void set_boundary_zeros(ArrayPtr &ap, int3 lb, int3 rb);
    
    void set_boundary_zeros(ArrayPtr &ap, Box sub_box); 

    //inline int calc_id(int i, int j, int k, int3 S);

    void calc_inside(ArrayPtr &ap, ArrayPtr &A, int3 lbound, int3 rbound);

    void calc_outside(ArrayPtr &ap, ArrayPtr &B, int3 lbound, int3 rbound);

    // convert a mpi array to sequential array

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
    void set_ref_const(ArrayPtr& A, const Box& A_box, T val) {
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
      Box sub_box(rsx[idx * 3], rsx[idx * 3 + 1],
                  rsy[idy * 3], rsy[idy * 3 + 1], 
                  rsz[idz * 3], rsz[idz * 3 + 1]);

      ///:for i in ['int', 'float', 'double']
      if (A->get_data_type() == DATA_${i.upper()}$) {
        oa::internal::set_buffer_subarray_const<${i}$, T>(
          (${i}$*) A->get_buffer(),
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

    void set_with_mask(ArrayPtr& A, const Box& sub_box, const ArrayPtr& B, const ArrayPtr& mask);
    void set_with_mask(ArrayPtr& A, const ArrayPtr& B, const ArrayPtr& mask);

    // rep arry_B = rep_A(arryA, 1, 2, 3)
    ArrayPtr rep(ArrayPtr& A, int x, int y, int z);

    //ArrayPtr create_local_array(const Shape& gs, DataType dt);

    template<class T>
    ArrayPtr create_local_array(const Shape& gs, T* buf){
      int sw = Partition::get_default_stencil_width();
      DataType dt = oa::utils::to_type<T>();        
      ArrayPtr ap = zeros(MPI_COMM_SELF, gs, sw, dt);
      T* dst_buf = (T*)ap->get_buffer();

      const int xs = 0;
      const int xe = gs[0] + 2 * sw;
      const int ys = 0;
      const int ye = gs[1] + 2 * sw;
      const int zs = 0;
      const int ze = gs[2] + 2 * sw;

      const int M = xe;
      const int N = ye;
      const int P = ze;
      
      for(int k = zs + sw; k < ze - sw; k++){
        for(int j = ys + sw; j < ye - sw; j++){
          for(int i = xs + sw; i < xe - sw; i++){
            dst_buf[i+j*M+k*M*N] =
              buf[i-sw + (j-sw) * gs[0] + (k-sw)*gs[0]*gs[1]];
          }
        }
      }
      return ap;
    }


    template<class T>
    void set(ArrayPtr& A, const Box& ref_box,
            T* buf, Shape& buf_shape){

      int num = buf_shape[0] * buf_shape[1] * buf_shape[2];
      for(int i = 0; i < num; ++i){
        std::cout<<" "<< buf[i] << " ";
      }
      
      assert(ref_box.shape() == buf_shape);
      
      Box local_box = A->get_local_box();
      Box local_ref_box = local_box.get_intersection(ref_box);

      int3 offset_local =
        local_ref_box.starts() - local_box.starts();
      int x1 = offset_local[0];
      int y1 = offset_local[1];
      int z1 = offset_local[2];
      
      int3 offset_ref =
        local_ref_box.starts() - ref_box.starts();
      int x2 = offset_ref[0];
      int y2 = offset_ref[1];
      int z2 = offset_ref[2];
      
      Shape slr = local_ref_box.shape();
      int M = slr[0];
      int N = slr[1];
      int P = slr[2];

      Shape bs = A->buffer_shape();

      const int sw = A->get_partition()->get_stencil_width();

      void* dst_buf = A->get_buffer();
      
      switch(A->get_data_type()){
        ///:for t in ['int', 'float', 'double']
      case (DATA_${t.upper()}$):
        for(int k = 0; k < P; ++k){
          for(int j = 0; j < N; ++j){
            for(int i = 0; i < M; ++i){
              int idx1 = (i + sw + x1) +
                (j + sw + y1) * bs[0] +
                (k + sw + z1) * bs[0] * bs[1];
              int idx2 = (i + x2) +
                (j + y2) * buf_shape[0] +
                (k + z2) * buf_shape[0] * buf_shape[1];
              ((${t}$*)dst_buf)[idx1] = buf[idx2];
            }
          }
        } 
        break;
      ///:endfor
      }
    }

    ArrayPtr make_psudo3d(const ArrayPtr& B);


  }
}

#endif
