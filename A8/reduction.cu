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
__device__ int sum;
//__constant__ int nthreads = 512;

__global__ void reduction1(int* arr){
    __shared__ float partialSum[512];
    
    int i = threadIdx.x;

    for(int k=i; k<blockDim.x; k++)
        partialSum[k] = arr[k];
    
    for (int stride = 1; stride<blockDim.x; stride *= 2){
        if (i % (2 * stride) == 0)
            partialSum[i] += partialSum[i + stride];
        __syncthreads();
    }
    sum += partialSum[0];

}

int main() {
    checkForGPU();
    int size = pow(2,24);
    //int size = 10;
    int nbytes = size * sizeof(int);
    int *arr = (int*) malloc(nbytes);
    int *d_arr;
    int nthreads = 512;
    int nblocks = (size-1)/nthreads + 1;
    
    srand(time(NULL));
    for(int i=0; i<size; i++)
        arr[i] = rand() % 256;

    cudaMalloc(&d_arr, nbytes);
    cudaMemcpy(d_arr, arr, nbytes, cudaMemcpyHostToDevice);

    dim3 gridSize(nblocks,1,1);
    dim3 blockSize(nthreads,1,1);
    
    //cudaMallocManaged(arr, nbytes); 

    reduction1<<<gridSize,blockSize>>>(d_arr);

    cudaMemcpy(arr, d_arr, nbytes, cudaMemcpyDeviceToHost);
    

    // for(int i=0; i<size; i++)
    //     printf("%d ", arr[i]); 

    printf("\nSum = %d", sum);
    free(arr);    
    cudaFree(d_arr);

    return 0;
}