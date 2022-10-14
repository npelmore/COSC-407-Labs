#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void *memcpy(void *dest, const void * src, size_t n);
void count_sort(int[],int);

int main (int argc, char *argv[]) {

	int n = 10000;
	int a[n];

	//creating the array with unordered integers
	for (int i = 0; i < n; i++){
		if (i % 2 == 0){
			a[i] = i*5;
		}
		else
			a[i] = i+3;
	}

	double t = omp_get_wtime();

	//sort array
	count_sort(a,n);

	t = 1000 * (omp_get_wtime() - t);

	//print part of sorted array to confirm it worked
	for (int i = 0; i < 50; i++)
		printf("%d ", a[i]);

	printf("\nFinished sorting in %.1f ms.\n", t);

	return 0;
}

void count_sort(int a[], int n) {
	int i, j, count;
	int* temp = malloc(n * sizeof(int));

#pragma omp parallel for num_threads(8) private(j,count)
	for (i = 0; i < n; i++){
		//count all elements < a[i]
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j]<a[i] ||(a[j]==a[i] && j<i))
				count++;
		//place a[i] at right order
		temp[count] = a[i];
	}

	memcpy(a, temp, n * sizeof(int));
	free(temp);
}

//Sequential: Finished sorting in 170.0 ms.
//Parallel: Finished sorting in 37.0 ms.

//a) j and count need to be private to keep each thread using their own j in the inner for loop, and their own count while sorting. The temporary array and original array needs to be shared.
//b) because we are using a temporary array and a private count variable, there are no dependencies. Concurrent threads are not accessing and writing to shared data at the same time.


