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

  int numThreads, tid;

  /* This creates a team of threads; each thread has own copy of variables  */
#pragma omp parallel private(numThreads, tid)
 {


 }
 return 0;
}


