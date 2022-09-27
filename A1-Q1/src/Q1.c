/*
 ============================================================================
 Name        : Q1.c
 Author      : Nathan Pelmore
 Description : First Lab Assignment
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main() {
	int n1, n2, n3, n4, num;
	double sum, avg;
	printf("Enter 4 integers separated by spaces:\n");
	fflush(stdout);

	scanf("%d %d %d %d", &n1, &n2, &n3, &n4);
	int arr[] = {n1, n2, n3, n4};
	num = 0;

	for(int i=0; i<4; i++){
		sum+=arr[i];
	}

	//sum = n1+n2+n3+n4;
	avg = sum/4;

	for(int i=0; i<4; i++){
		if(arr[i]>avg){
			num++;
		}
	}
	if(num==1) {
		printf("There is %d entry above the average (%1.1f)", num, avg);
	}
	else {
		printf("There are %d entries above the average (%1.1f)", num, avg);
	}
	return 0;
}
