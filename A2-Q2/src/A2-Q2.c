/*
 ============================================================================
 Name        : A2-Q2.c
 Author      : Nathan Pelmore
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
	clock_t start_t, end_t;
	double total_t;
	int* addVec2(int* A, int* B, int size);
	int size = 50000000;
	int* A = calloc(size, sizeof(int));
	int* B = calloc(size, sizeof(int));

	if(A==NULL || B==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		exit(EXIT_FAILURE);
	}

	start_t = clock();
	int* C = addVec2(A,B,size);
	end_t = clock();

	total_t = (double)(end_t - start_t);

	for(int i=0;i<10;i++){
		printf("%d ", C[i]);
	}
	printf("\nExecution Time: %1.1f ms", total_t);

	free(A);
	free(B);
	free(C);
	return EXIT_SUCCESS;
}

int* addVec2(int* A, int* B, int size){
	int* C = calloc(size, sizeof(int));
	if(C==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		exit(EXIT_FAILURE);
	}
	for(int i=0;i<size;i++){
			C[i] = A[i] + B[i];
	}
	return C;
}

/*
Result:
50 Million = 364.0 ms

 */
