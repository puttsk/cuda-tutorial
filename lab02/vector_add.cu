#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N (1<<20)
#define MAX_ERR 1e-6

double gettime(){
    struct timeval t;
    gettimeofday(&t, 0);
    return (t.tv_sec * 1000000 + t.tv_usec);
}

__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (tid < n){
    //    for(int i = threadIdx.x; i < n; i += blockDim.x)
        out[tid] = a[tid] + b[tid];
    }
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    //a   = (float*)malloc(sizeof(float) * N);
    //b   = (float*)malloc(sizeof(float) * N);
    //out = (float*)malloc(sizeof(float) * N);

    // Allocate device memory 
    cudaMallocManaged((void**)&a, sizeof(float) * N);
    cudaMallocManaged((void**)&b, sizeof(float) * N);
    cudaMallocManaged((void**)&out, sizeof(float) * N);

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Timing kernel execution
    double timer = -gettime();

    // Transfer data from host to device memory
    //cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = ((N + block_size) / block_size);
    vector_add<<<grid_size,block_size>>>(out, a, b, N);
    
    cudaDeviceSynchronize();
    //cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
    timer += gettime();

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED: %f s\n", timer / 1e6);

    cudaFree(a);
    cudaFree(b);
    cudaFree(out);

    //free(a); 
    //free(b); 
    //free(out);
}
