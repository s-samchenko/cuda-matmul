#include <cuda_runtime.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int nthreads = (BM / TM) * (BN / TN);

__global__ void matmul_blockedv2_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];
    float acc[TM][TN] = {}; // 8*8 = 64

    for (int j = 0; j < N; j += BK) {
        for (int i = tid; i < BM * BK; i += nthreads){
            As[i % BK][i / BK] = A[(row + i/BK) * N + (j + i%BK)];
        }
        for (int i = tid; i < BK * BN; i += nthreads){
            Bs[i / BN][i % BN] = B[(j + i/BN) * N + (col + i%BN)];
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k){
            float a[TM], b[TN];
            for (int m = 0; m < TM; ++m) a[m] = As[k][ty * TM + m];
            for (int n = 0; n < TN; ++n) b[n] = Bs[k][tx * TN + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++) {
            int r = row + ty * TM + m;
            int c = col + tx * TN + n;
            if (r < N && c < N)
                C[r * N + c] = acc[m][n];
        }
}

void matmul_blockedv2_launch(const float* A, const float* B, float* C, int N) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    matmul_blockedv2_kernel<<<grid, block>>>(A, B, C, N);
}
