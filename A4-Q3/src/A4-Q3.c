#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */
int main (int argc, char *argv[]) {
	int n = 10;
	int a[n];
  a[0] = 0;
  int i;

  /* This creates a team of threads; each thread has own copy of variables  */
#pragma omp parallel for
	  for(i=1; i<n; i++){
		  a[i] = i*(i+1)/2;  //Triangular numbers sequence formula
	  }

	  for(i=0; i<n; i++){
	 		  printf("%d ", a[i]);
	 	  }

 return 0;
}

//The only reasonable way to parallelize this loop is to change the formula of the current code.
//The the product of the unparallelizable code is an array of the triangular number sequence.
//All we need to do is get rid of the dependency by changing the formula to that of the triangular numbers sequence formula.
//With that change, the code is parallelizable, as seen above.

