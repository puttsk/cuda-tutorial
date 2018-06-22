#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cudaHello(){
    printf("Hello World with NVCC!\n");
}

int main() {
    cudaHello<<<1,1>>>(); 
    return 0;
}
