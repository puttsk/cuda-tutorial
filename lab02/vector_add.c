#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>

#define N (1<<20)
#define MAX_ERR 1e-6

double gettime(){
    struct timeval t;
    gettimeofday(&t, 0);
    return (t.tv_sec * 1000000 + t.tv_usec);
}

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // profiling 
    double timer = -gettime();
    vector_add(out, a, b, N);
    timer += gettime();

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    printf("PASSED: %f ms\n", timer / 1e3);
}
