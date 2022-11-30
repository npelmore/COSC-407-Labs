%%cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "device_launch_parameters.h"

#define CHK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("Error%d: %s:%d\n",err,__FILE__,__LINE__); printf(cudaGetErrorString(err)); cudaDeviceReset(); exit(1); }}

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

__global__ void reduction1(float* arr, float* partialSums){
    __shared__ float partialSum[512];

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;
    partialSum[i] = arr[x];
    __syncthreads();
    
    for (int stride = 1; stride<blockDim.x; stride *= 2){
    //for (int stride = 1; stride<blockDim.x; stride <<= 1){ // reduction with << instead of *
        if (i % (2 * stride) == 0)
            partialSum[i] += partialSum[i + stride];
        __syncthreads();
    }
    if(i==0)
        partialSums[blockIdx.x] += partialSum[0];
}

__global__ void reduction2(float* arr, float* partialSums){
    __shared__ float partialSum[512];

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;
    partialSum[i] = arr[x];
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride >= 1; stride = stride/2){
        if (i<stride)
            partialSum[i] += partialSum[i + stride];
        __syncthreads();
    }
    if(i==0)
        partialSums[blockIdx.x] += partialSum[0];
}

__global__ void reduction3(float* arr){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;

    for (int stride = 1; stride<blockDim.x; stride *= 2){
        if (i % (2 * stride) == 0)
            arr[x] += arr[x + stride];
        __syncthreads();
    }
}

__global__ void reduction4(float* arr){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;

    for (int stride = blockDim.x/2; stride >= 1; stride = stride/2){
        if (i < stride)
            arr[x] += arr[x + stride];
        __syncthreads();
    }
}

int main() {
    checkForGPU();
    
    const int size = 1<<24;
    int nbytes = size * sizeof(float);
    float* mainArr = (float*) malloc(nbytes);
    float* arr = (float*) malloc(nbytes);
    float* d_arr;
    
    CHK(cudaMalloc(&d_arr, nbytes));
    
    srand(time(NULL));
    for(int i=0; i<size; i++)
        //mainArr[i] = 1;
        mainArr[i] = rand() % 256;

    for(int i=0; i<size; i++)
        arr[i] = mainArr[i];

    CHK(cudaMemcpy(d_arr, arr, nbytes, cudaMemcpyHostToDevice));

    int nthreads = 512;
    int nblocks = (size-1)/nthreads + 1;
    float sum = 0;
    dim3 gridSize(nblocks,1,1);
    dim3 blockSize(nthreads,1,1);
    float* partialSums;
   

    printf("Reducing an array of %d floats on a grid of (%d,1,1) blocks, each block with (%d,1,1) threads\n", size, nblocks, nthreads);

    //reduction 1------------------------------------------------------------------------------------------------------------------------
    CHK(cudaMallocManaged(&partialSums, nblocks));
   
    float t = clock();
    reduction1<<<gridSize,blockSize>>>(d_arr, partialSums);
    CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
 
    for (int i=0; i<nblocks; i++){
        sum += partialSums[i];
    }
    
    printf("Using shared memory, More divergence: GPU time: %.3f ms GPU sum: %.2f\n", t, sum);
    
    cudaFree(partialSums);
    sum = 0;
    
    //reduction 2------------------------------------------------------------------------------------------------------------------------
    CHK(cudaMallocManaged(&partialSums, nblocks));

    t = clock();
    reduction2<<<gridSize,blockSize>>>(d_arr, partialSums);
    CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;

    for (int i=0; i<nblocks; i++){
        sum += partialSums[i];
    }
    
    printf("Using shared memory, Less divergence: GPU time: %.3f ms GPU sum: %.2f\n", t, sum);
    
    cudaFree(partialSums);
    sum = 0;

    //reduction 3------------------------------------------------------------------------------------------------------------------------
    t = clock();
    reduction3<<<gridSize,blockSize>>>(d_arr);
    CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;

    CHK(cudaMemcpy(arr, d_arr, nbytes, cudaMemcpyDeviceToHost));
    for (int i=0; i<size; i+=size/nblocks){
        sum += arr[i];
    }

    printf("Using global memory, More divergence: GPU time: %.3f ms GPU sum: %.2f\n", t, sum);
   
    for(int i=0; i<size; i++)
        arr[i] = mainArr[i];

    CHK(cudaMemcpy(d_arr, arr, nbytes, cudaMemcpyHostToDevice));
    sum = 0;

    //reduction 4------------------------------------------------------------------------------------------------------------------------
    t = clock();
    reduction4<<<gridSize,blockSize>>>(d_arr);
    CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;

    CHK(cudaMemcpy(arr, d_arr, nbytes, cudaMemcpyDeviceToHost));
    for (int i=0; i<size; i+=size/nblocks){
        sum += arr[i];
    }

    printf("Using global memory, Less divergence: GPU time: %.3f ms GPU sum: %.2f\n", t, sum);

    //Free malloced and cudaMalloced variables------------------------------------------------------------------------------------------
    free(mainArr);    
    free(arr);    
    cudaFree(d_arr);
    
    return 0;
}

/*
Output:
Reducing an array of 16777216 floats on a grid of (32768,1,1) blocks, each block with (512,1,1) threads
Using shared memory, More divergence: GPU time: 3.813 ms GPU sum: 2139051520.00
Using shared memory, Less divergence: GPU time: 2.186 ms GPU sum: 2139051520.00
Using global memory, More divergence: GPU time: 3.423 ms GPU sum: 2139051520.00
Using global memory, Less divergence: GPU time: 1.651 ms GPU sum: 2139051520.00

Difference between * and << on reduction1:
* GPU time: 3.813 ms
<< GPU time: 3.719 ms
Difference: << is 0.094ms faster than *
*/