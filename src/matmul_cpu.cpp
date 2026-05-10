#include <stdio.h>
#include <cmath>

void matmul_cpu(const float* a, const float* b, float* c, int N){
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            c[i*N + j] = 0.f;
            for (int k = 0; k < N; ++k){
                c[i*N + j] += a[i*N + k] * b[k*N + j];
            }
        }
    }
}

bool verify(const float *ref, const float* gpu, int N){
    for (int i = 0; i < N*N; ++i){
        if (fabsf(ref[i] - gpu[i]) > 1e-3f) {
            fprintf(stderr, "Mismatch at index %d: CPU=%.6f GPU=%.6f\n",
                    i, ref[i], gpu[i]);
            return false;
        }
    }

    return true;
}