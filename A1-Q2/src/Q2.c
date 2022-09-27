/*
 ============================================================================
 Name        : A1-Q2.c
 Author      : Nathan Pelmore
 Description : Q2
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
	clock_t start_t, end_t;
	double total_t;
	int n = 1000000;
	//int n = 10000000;
	//int n = 50000000;
	//int n = 200000000;
	int sum = 0;
	int* A = (int*) malloc(n * sizeof(int));
	int* B = (int*) malloc(n * sizeof(int));
	int* C = (int*) malloc(n * sizeof(int));

	if(A==NULL || B==NULL || C==NULL){
		fprintf(stderr, "Not Enough Memory\n");
		exit(EXIT_FAILURE);
	}

	start_t = clock();
	for(int i=0; i<n; i++){
		A[i] = i*3;
		B[i] = -i*3;
		C[i] = 	A[i]+B[i];
		sum += C[i];
	}
	end_t = clock();

	total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;

	printf("Sum: %d \nExecution Time: %1.2f sec", sum, total_t);

	exit(EXIT_SUCCESS);
}
/*
Results:
1 million = 0.02 seconds
10 million = 0.16 seconds
50 million = 0.68 seconds
200 million = Not Enough Memory
*/
