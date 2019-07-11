subroutine wave(nt , nx, ny)
     implicit none
     type(array) :: u_new, u, u_old
     integer,intent(in) :: nx, ny, nt
     u = seqs(nx, ny, 1 dt=OA_DOUBLE)
     u_new = seqs(nx, ny, 1, dt=OA_DOUBLE)
     u_old = seqs(nx, ny, 1, dt=OA_DOUBLE)
     do n=1,nt
        u_new = 2 * ( sigma^2*(DXF(u)-DXB(u)) + gamma**2*(DYF(u)-DYB(u)) ) + 2u - u_old
        u_old = u
        u = u_new
     enddo
   end subroutine

subroutine Runge_Kutte(nt, nx, ny)
  implicit none
  type(array) :: T, Tk
  double precision :: dx, dy,dt, alpha
  integer , intent(in):: nt, nx,ny
  integer :: k
  T = seqs(nx, ny, 1, dt=OA_DOUBLE)
  Tk = seqs(nx, ny, 1, dt=OA_DOUBLE)
  k1 = seqs(nx, ny, 1, dt=OA_DOUBLE)
  k2 = seqs(nx, ny, 1, dt=OA_DOUBLE)
   do k = 1,nt
     k1=2*alpha* ( (DXF(T)-DXB(T))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
     Tk=T+k1*dt
     k2=2*alpha* ( (DXF(Tk)-DXB(Tk))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
     T=T+dt/2*(k1+k2)
   enddo
 end subroutine



subroutine eular(nt,nx,ny)
    implicit none
    type(array):: T
    double precision ::  dt, dx, dy, alpha 
    integer, intent(in):: nx,ny,nt
    integer :: k
    dx = 0.1
    dy = 0.1
    dt = 0.1
    alpha = 0.1

    T = seqs(nx,ny,1, dt=OA_DOUBLE)
    do  k=1,nt
        T = T + dt*alpha*2* ( (DXF(T)-DXB(T))/(dx**2) + (DYF(T)-DYB(T))/(dy**2) )
    enddo
end subroutine

subroutine height(nt, nx, ny)
  implicit none
  type(array) :: D, U, V 
  double precision :: dt
  integer,intent(in) :: nx,ny,nt
  integer :: i
  dt = 0.1
  D = seqs(nx,ny,1, dt=OA_DOUBLE)
  U = seq(nx,ny,1,dt=OA_DOUBLE)
  V = seq(nx,ny,1,dt=OA_DOUBLE)
  E = seqs(nx,ny,1,dt=OA_DOUBLE)
  do i=1,nt
    E=E-dt*(DXF(AXB(D)*U)+DYF(AYB(D)*V))
  enddo
end subroutine
