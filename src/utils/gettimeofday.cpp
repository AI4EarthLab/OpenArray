#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

extern "C"{
  

  void gettimeofday1(int *ierr2, long long* val)
  {
    struct timeval time1;
    int ierr;
  
    ierr = gettimeofday(&time1, NULL);
    *ierr2 = ierr;

    if(ierr != 0)
      printf("bad return of gettimeofday, ierr = %d \n", ierr);
  
    /* tim[0] = time1.tv_sec;  */
    /* tim[1] = time1.tv_usec; */
    *val = (time1.tv_sec * 1000000 + time1.tv_usec);
  }

}
