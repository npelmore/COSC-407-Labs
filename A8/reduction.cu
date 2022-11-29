%%cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "device_launch_parameters.h"

void checkForGPU() {
    // This code attempts to check if a GPU has been allocated.
    // Colab notebooks without a GPU technically have access to NVCC and will
    // compile and execute CPU/Host code, however, GPU/Device code will silently
    // fail. To prevent such situations, this code will warn the user.
    int count;
    cudaGetDeviceCount(&count);
    if (count <= 0 || count > 100) {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("->WARNING<-: NO GPU DETECTED ON THIS COLLABORATE INSTANCE.\n");
        printf("IF YOU ARE ATTEMPTING TO RUN GPU-BASED CUDA CODE, YOU SHOULD CHANGE THE RUNTIME TYPE!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
}

int main() {
    checkForGPU();
    //int size = pow(2,24);
    int size = 10;
    int* arr = (int*) malloc(size * sizeof(int));

    srand(time(NULL));
    for(int i=0; i<size; i++)
        arr[i] = rand() % 256;
    





    // for(int i=0; i<size; i++)
    //     printf("%d ", arr[i]); 

    free(arr);    

    return 0;
}