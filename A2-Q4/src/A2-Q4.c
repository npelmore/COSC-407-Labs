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

	//numThreads = 4;

	start_t = clock();
	int* A = vecCreate(size);
	end_t = clock();

	total_t = (double)(end_t - start_t);

	printf("Using Serial Code\nv[%d] = %d\nTime: %1.2f ms\n\n", (sizeof(*A)/sizeof(A[0])) , size, total_t);


	start_t = clock();
	int* B = vecCreateOpenMP(size,numThreads);
	end_t = clock();

	total_t = (double)(end_t - start_t);


	return 0;
}

int* vecCreate (int size){
	int* A = malloc(size * sizeof(int));
	if(A==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		exit(EXIT_FAILURE);
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
		exit(EXIT_FAILURE);
	}

#pragma omp parallel num_threads(num_thread)
	{

	}
}


