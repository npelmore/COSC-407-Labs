/*
 ============================================================================
 Name        : practice.c
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
	int n = 10;
	int a[n];
  a[0] = 0;
  int i;

  /* This creates a team of threads; each thread has own copy of variables  */
#pragma omp parallel for
	  for(i=1; i<n; i++){
		  a[i] = i*(i+1)/2;
	  }

	  for(i=0; i<n; i++){
	 		  printf("%d ", a[i]);
	 	  }

 return 0;
}


