# Lab 01: Say Hello to CUDA

## Introduction 

This tutorial is an introduction for writing your first CUDA C program. 

## A quick comparison between CUDA and C

Following table compares a hello world program in C and CUDA side-by-side. 

<table>
<tr><td> <b>C</b> </td><td> <b>CUDA</b> </td></tr>
<tr>
<td>

```C
void c_hello(){
    printf("Hello World!\n");
}

int main() {
    c_hello()
    return 0;
}
```

</td>
<td>

```C
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}
```

</td>
</tr>
</table>

The major difference between C and CUDA implementation is `__global__` specifier and `<<<...>>>` syntax. The ```__global__``` specifier indicates a function that runs on device (GPU). The function can be called through host code, e.g. the `main()` function in the example. The function is also known as "*kernels*". 

When a kernel is called, its execution configuration is provided through `<<<...>>>` syntax, e.g. `cuda_hello<<<1,1>>>()` from the example. In CUDA term, this is called "*kernel launch*". We will discuss about the parameter `(1,1)` later in this tutorial. 

## Compiling CUDA programs

Compiling a CUDA program is similar to C program. NVIDIA provides a CUDA compiler called `nvcc` in the CUDA toolkit to compile CUDA code, typically stored in a file with extension `.cu`. For example

```bash
$> nvcc hello.cu -o hello
```

For the latest version of `nvcc`, you might see following warning when compiling a CUDA program using above command

```
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
```

This warning can be ignored as of now. 

## Putting things in actions. 

The CUDA hello world example does nothing, and even if the program is compiled, nothing will show up on screen. To get things into action, we will looks at vector addition. 

Following is an example vector addition implemented in C ([`./vector_add.c`](./vector_add.c)).

```C
#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);
}

```

## Exercise 1: Converting vector addition to CUDA

In the first exercise, we will convert `vector_add.c` to CUDA program `vector_add.cu` by using the hello world as example.

1. Copy `vector_add.c` to `vector_add.cu`

```bash
$> cp vector_add.c vector_add.cu
```

2. Convert `vector_add()` to GPU kernel

```C
__global__ void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}
```

3. Change `vector_add()` call in `main()` to kernel call

```C
vector_add<<<1,1>>>(out, a, b, N);
```

4. Compile and run the program
```bash
$> nvcc vector_add.c -o vector_add
$> ./vector_add
```

You will notice that the program does not work correctly.

## Acknowledgments

* Contents are adopted from [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/) by Mark Harris, NVIDIA and [CUDA C/C++ Basics](http://www.int.washington.edu/PROGRAMS/12-2c/week3/clark_01.pdf) by Cyril Zeller, NVIDIA. 


