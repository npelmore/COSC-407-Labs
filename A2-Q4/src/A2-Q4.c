/*
 ============================================================================
 Name        : A2-Q4.c
 Author      : Nathan Pelmore
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main (int argc, char *argv[]) {
	clock_t start_t, end_t;
	double total_t;
	int numThreads = 4;
	int size = 50000000;
	int* vecCreate (int size);
	int* vecCreateOpenMP(int size, int num_thread);


	start_t = clock();
	int* A = vecCreate(size);
	end_t = clock();
	total_t = (double)(end_t - start_t);

	if(A!=NULL){

	printf("Using Serial Code\nv[%d] = %d\nTime: %1.2f ms\n\n", size-1 , A[size-1], total_t);

	}

	start_t = clock();
	int* B = vecCreateOpenMP(size,numThreads);
	end_t = clock();
	total_t = (double)(end_t - start_t);

	if(B!=NULL){
	printf("Using OpenMP with %d threads:\nv[%d] = %d\nTime: %1.2f ms\n\n", numThreads, size-1 , B[size-1], total_t);
	}

	free(A);
	free(B);

	return 0;
}

int* vecCreate (int size){
	int* A = malloc(size * sizeof(int));
	if(A==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		return NULL;
	}

	for(int i=0;i<size;i++){
		A[i]=i;
	}
	return A;
}

int* vecCreateOpenMP(int size, int num_thread){

	if(size % num_thread != 0){
		fprintf(stderr, "Error: number of threads must be divisible by vector size\n");
		exit(EXIT_FAILURE);
	}

	int* B = malloc(size * sizeof(int));
	if(B==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		return NULL;
	}

#pragma omp parallel num_threads(num_thread)
	{
		int id = omp_get_thread_num();
		int sizeOfSection = size/num_thread;
		int start = id * sizeOfSection;
		int end = start + sizeOfSection;
		for(int i=start;i<end;i++){
			B[i]=i;
		}
	}
	return B;

}

/*
Successful:
	Using Serial Code
	v[49999999] = 49999999
	Time: 101.00 ms

	Using OpenMP with 4 threads:
	v[49999999] = 49999999
	Time: 61.00 ms

num_thread not divisible by size:
	Using Serial Code
	v[49999999] = 49999999
	Time: 82.00 ms

	Error: number of threads must be divisible by vector size

Unsuccessful memory allocation:
	Not Enough Memory
	Not Enough Memory
 */



