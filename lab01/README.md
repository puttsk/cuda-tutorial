# Say Hello to CUDA: Writing your first CUDA program

## Introduction 

### Hello World C and CUDA side-by-side
<table>
<tr><td> C </td><td> CUDA </td></tr>
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



### Vector Addition in C

```C
#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

void main(){
    a = (float*)malloc(sizeof(float) * N);
    …
    vector_add(out, a, b, N);
    …
    free(a);
}
```

## Exercise 1: Converting vector addition to CUDA



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



Contents are adopted from [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/) by Mark Harris, NVIDIA and [CUDA C/C++ Basics](http://www.int.washington.edu/PROGRAMS/12-2c/week3/clark_01.pdf) by Cyril Zeller, NVIDIA. 


