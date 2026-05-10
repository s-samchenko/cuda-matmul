#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    __shared__ float TileA[TILE][TILE];
    __shared__ float TileB[TILE][TILE];

    float sum = 0.f;

    for (int t = 0; t < N / TILE; ++t) {
        TileA[ty][tx] = A[row * N + (t * TILE + tx)];
        TileB[ty][tx] = B[(t * TILE + ty) * N + col];
        __syncthreads();

        for (int i = 0; i < TILE; ++i)
            sum += TileA[ty][i] * TileB[i][tx];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void matmul_tiled_launch(const float* A, const float* B, float* C, int N) {
    dim3 blockDim(TILE, TILE);
    dim3 gridDim((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_tiled_kernel<<<gridDim, blockDim>>>(A, B, C, N);
}
