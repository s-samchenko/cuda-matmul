#include <cuda_runtime.h>
#include <cuda_pipeline.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 8;
constexpr int TN = 8;
constexpr int nthreads = (BM / TM) * (BN / TN);

__global__ void matmul_pipelined_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int N){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;

    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];
    float acc[TM][TN] = {};

    int buf = 0;

    // prefill tile 0 into buf 0
    for (int i = tid; i < BM * BK / 4; i += nthreads){
        int row_in_tile = i / (BK / 4);
        int col4 = i % (BK / 4);
        float4 tmp = *reinterpret_cast<const float4*>(
                &A[(row + row_in_tile) * N + col4 * 4]
        );
        As[buf][col4 * 4 + 0][row_in_tile] = tmp.x;
        As[buf][col4 * 4 + 1][row_in_tile] = tmp.y;
        As[buf][col4 * 4 + 2][row_in_tile] = tmp.z;
        As[buf][col4 * 4 + 3][row_in_tile] = tmp.w;
    }
    for (int i = tid; i < BK * BN / 4; i += nthreads){
        int row_in_tile = i / (BN / 4);
        int col4        = i % (BN / 4);
        __pipeline_memcpy_async(
            &Bs[buf][row_in_tile][col4 * 4],
            &B[row_in_tile * N + (col + col4 * 4)],
            16
        );
    }
    __pipeline_commit();

    for (int j = BK; j < N; j += BK) {
        buf ^= 1;

        for (int i = tid; i < BM * BK / 4; i += nthreads){
            int row_in_tile = i / (BK / 4);
            int col4        = i % (BK / 4);
            float4 tmp = *reinterpret_cast<const float4*>(
                    &A[(row + row_in_tile) * N + (j + col4 * 4)]
            );
            As[buf][col4 * 4 + 0][row_in_tile] = tmp.x;
            As[buf][col4 * 4 + 1][row_in_tile] = tmp.y;
            As[buf][col4 * 4 + 2][row_in_tile] = tmp.z;
            As[buf][col4 * 4 + 3][row_in_tile] = tmp.w;
        }

        for (int i = tid; i < BK * BN / 4; i += nthreads){
            int row_in_tile = i / (BN / 4);
            int col4        = i % (BN / 4);
            __pipeline_memcpy_async(
                &Bs[buf][row_in_tile][col4 * 4],
                &B[(j + row_in_tile) * N + (col + col4 * 4)],
                16
            );
        }
        __pipeline_commit();

        // wait for Bs[1-buf] (committed previous iteration)
        __pipeline_wait_prior(1);
        __syncthreads();

        for (int k = 0; k < BK; ++k){
            float a[TM], b[TN];
            for (int m = 0; m < TM; ++m) a[m] = As[1 - buf][k][ty * TM + m];
            for (int n = 0; n < TN; ++n) b[n] = Bs[1 - buf][k][tx * TN + n];
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a[m] * b[n];
        }

        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    for (int k = 0; k < BK; ++k){
        float a[TM], b[TN];
        for (int m = 0; m < TM; ++m) a[m] = As[buf][k][ty * TM + m];
        for (int n = 0; n < TN; ++n) b[n] = Bs[buf][k][tx * TN + n];
        for (int m = 0; m < TM; ++m)
            for (int n = 0; n < TN; ++n)
                acc[m][n] += a[m] * b[n];
    }

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++) {
            int r = row + ty * TM + m;
            int c = col + tx * TN + n;
            if (r < N && c < N)
                C[r * N + c] = acc[m][n];
        }
}

void matmul_pipelined_launch(const float* A, const float* B, float* C, int N) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    matmul_pipelined_kernel<<<grid, block>>>(A, B, C, N);
}
