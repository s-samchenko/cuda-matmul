#include <cuda_runtime.h>
#include <cstdlib>
#include <string>
#include <vector>
#include "benchmark.cuh"
#include "matmul_cpu.h"
#include "kernels.cuh"

int main() {
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    printf("GPU: %s | SMs: %d | VRAM: %.0f MB\n\n",
           props.name,
           props.multiProcessorCount,
           (float)props.totalGlobalMem / (1024 * 1024));

    std::vector<KernelEntry> kernels = {
        {"naive",    matmul_naive_launch},
        {"tiled",    matmul_tiled_launch},
        {"blocked",  matmul_blocked_launch},
        {"blockedv2",  matmul_blockedv2_launch},
        {"vec4",  matmul_vec4_launch}
    };

    int sizes[] = {256, 512, 1024, 2048};

    printf("%-20s | %-7s | %-10s | %-12s | %-14s | %s\n",
           "Kernel", "N", "ms/run", "GB/s", "GFLOPS", "Bound");
    printf("%s\n", std::string(85, '-').c_str());

    for (auto& k : kernels) {
        // Correctness check at small N before benchmarking
        {
            constexpr int N = 256;
            size_t bytes = N * N * sizeof(float);

            float* h_A   = (float*)malloc(bytes);
            float* h_B   = (float*)malloc(bytes);
            float* h_C   = (float*)malloc(bytes);
            float* h_ref = (float*)malloc(bytes);

            for (int i = 0; i < N * N; i++) {
                h_A[i] = (float)rand() / RAND_MAX;
                h_B[i] = (float)rand() / RAND_MAX;
            }
            matmul_cpu(h_A, h_B, h_ref, N);

            float *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, bytes));
            CUDA_CHECK(cudaMalloc(&d_B, bytes));
            CUDA_CHECK(cudaMalloc(&d_C, bytes));
            CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));

            k.launch(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

            printf("[%s] Correctness: %s (N=%d)\n\n",
                   k.name, verify(h_ref, h_C, N) ? "PASSED" : "FAILED", N);

            free(h_A); free(h_B); free(h_C); free(h_ref);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }

        // Benchmark at all sizes
        for (int N : sizes) {
            size_t bytes = (size_t)N * N * sizeof(float);

            float *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, bytes));
            CUDA_CHECK(cudaMalloc(&d_B, bytes));
            CUDA_CHECK(cudaMalloc(&d_C, bytes));
            CUDA_CHECK(cudaMemset(d_C, 0, bytes));

            float ms = benchmark_kernel([&]() {
                k.launch(d_A, d_B, d_C, N);
            });

            print_metrics(k.name, N, ms);

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }
    }

    return 0;
}
