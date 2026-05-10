#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    if (row < N && col < N){
        for (int i = 0; i < N; ++i){
            sum += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = sum;
    }
}

void matmul_naive_launch(const float* A, const float* B, float* C, int N) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);
    matmul_naive_kernel<<<gridDim, blockDim>>>(A, B, C, N);
}
