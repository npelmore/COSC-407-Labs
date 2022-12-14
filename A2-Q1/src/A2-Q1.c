/*
 ============================================================================
 Name        : A2-Q1.c
 Author      : Nathan Pelmore
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
	clock_t start_t, end_t;
	double total_t;

	void addVec(int* C, int* A, int* B, int size);

	int size = 50000000;
	int* A = calloc(size, sizeof(int));
	int* B = calloc(size, sizeof(int));
	int* C = calloc(size, sizeof(int));

	if(A==NULL || B==NULL || C==NULL){
			fprintf(stderr, "Not Enough Memory\n");
			exit(EXIT_FAILURE);
	}

	start_t = clock();
	addVec(C,A,B,size);
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

void addVec(int* C, int* A, int* B, int size){
	for(int i=0;i<size;i++){
		C[i] = A[i] + B[i];
	}
}

/*
Result:
50 Million = 413.0 ms

 */
