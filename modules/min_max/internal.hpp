

namespace oa{
  namespace internal{

///:for k in [['max','>',''],['min','<',''],['abs_max','>'],['abs_min','<']]
///:set name = k[0]
///:set sy = k[1]
template<typename T>
void buffer_${name}$_const(T &val, int* pos, T *A, Box box, int sw) {
  int x = 0, y = 0, z = 0;
  int xs, xe, ys, ye, zs, ze;
  box.get_corners(xs, xe, ys, ye, zs, ze, sw);

  int M = xe - xs;
  int N = ye - ys;
  int K = ze - zs;

  //std::cout<<"MNK:"<<M<<" "<<N<<" "<<" "<<K<<" sw="<<sw<<std::endl;
      
  ///:mute
  ///:if k[0:3] == 'abs'
  ///:set op = "abs"
  ///:else
  ///:set op = ""
  ///:endif
  ///:endmute
      
  val = ${op}$(A[sw * M * N + sw * M + sw]);

  pos[0] = xs;
  pos[1] = ys;
  pos[2] = zs;

  //oa::utils::mpi_order_start(MPI_COMM_WORLD);
  for (int k = sw; k < K-sw; k++) {
    for (int j = sw; j < N-sw; j++) {
      for (int i = sw; i < M-sw; i++) {
        //printf("(%d,%d,%d) = %d"
        // std::cout<<"("<<i<<","<<j<<","<<k<<")="
        //          <<A[i + j * M + k * M * N]<<std::endl;
            
        if (A[i + j * M + k * M * N] ${sy}$ ${op}$(val)) {
          val = A[i + j * M + k * M * N];
          pos[0] = i - sw + xs;
          pos[1] = j - sw + ys;
          pos[2] = k - sw + zs;
        }
      }
    }
  }
  //oa::utils::mpi_order_end(MPI_COMM_WORLD);
                   }
///:endfor
}
}
