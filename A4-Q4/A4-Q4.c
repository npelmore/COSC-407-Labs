#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double my_drand(unsigned*);

int main (int argc, char *argv[]) {
	long long int number_of_tosses=1000000000;
		long long int number_in_circle = 0;
		double x,y, distance_squared;


		#pragma omp parallel reduction(+:number_in_circle) private(x,y,distance_squared)
		{
			unsigned int seed = omp_get_thread_num()+1;
			#pragma omp for
			for(long long int toss = 0; toss < number_of_tosses; toss++) {
				x = my_drand(&seed)*2-1;
				y = my_drand(&seed)*2-1;
				distance_squared = x * x + y * y;
				if (distance_squared <= 1)
					number_in_circle++;
			}
		}

		double piEstimate = 4*number_in_circle/((double) number_of_tosses);

		printf("Result: %f", piEstimate);

 return 0;
}


