%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void fillArray(double* a, int n){
  for (int i = 0; i < n; i++)
    a[i] = (double)i / n;
}
int main() {
  const int n = 10000000;
  double* a = (double *)malloc(n*sizeof(double));
  clock_t start_t, end_t;
	double total_t;

  start_t = clock();
  fillArray(a,n);
  end_t = clock();

  total_t = (double)(end_t - start_t);

  for (int i=0; i<5; i++)
    printf("a[%d]: %.7f\n", i, a[i]);
  printf("...\n");
  for (int i=n-6; i<n; i++)
    printf("a[%d]: %.7f\n", i, a[i]);

  printf("\nExecution Time: %1.1f ms", total_t);

  free(a);
  return(0);

}
//Execution Time: 88259.0 ms
