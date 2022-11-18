%%cu
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void fillArray(double* a, int n){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<n)
    a[i] = (double)i / n;
}
int main() {
  const int n = 10000000;
  double* a = (double *)malloc(n*sizeof(double));
  double *d_a;
  clock_t start_t, end_t;
	double total_t;
  cudaMalloc (&d_a, n*sizeof(double));
  int nThreads = 256;
  int nBlocks = n/nThreads;
  if (n%256) nBlocks++;

  start_t = clock();
  fillArray<<<nBlocks, nThreads>>>(d_a,n);
  end_t = clock();
  
  total_t = (double)(end_t - start_t);

  cudaMemcpy (a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i=0; i<5; i++)
    printf("a[%d]: %.7f\n", i, a[i]);
  printf("...\n");
  for (int i=n-6; i<n; i++)
    printf("a[%d]: %.7f\n", i, a[i]);

  printf("\nExecution Time: %1.1f ms", total_t);

  free(a);
  cudaFree(d_a);
  return(0); 
}
//Execution Time: 27.0 ms