/*
 ============================================================================
 Name        : A4-Q2.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */
int main (int argc, char *argv[]) {

#define NRA 20  /* number of rows in A */
#define NCA 30  /* number of columns in A = number of rows in B */
#define NCB 10  /* number of columns in matrix B */

	int A[NRA][NCA],B[NCA][NCB],C[NRA][NCB];
	int i,j,k;

#pragma omp parallel
	{
#pragma omp for private(j)
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCA;j++)
				A[i][j] = i +j;
#pragma omp for private(j)
		for (i = 0; i < NCA; i++)
			for (j = 0; j < NCB;j++)
				B[i][j] = i * j;
#pragma omp for private(j)
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCB;j++)
				C[i][j] = 0;

#pragma omp for private(j,k)
		for (i = 0; i < NRA; i++)
			for (j = 0; j < NCB;j++)
				for (k = 0; k < NCA; k++)
					C[i][j] += A[i][k] * B[k][j];
	}

	printf("Result n\n");
	for (i = 0; i < NRA; i++) {
		for (j = 0; j < NCB;j++)
			printf("%d ", C[i][j]);
		printf("\n");
	}

	return 0;
}


