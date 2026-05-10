#pragma once
#include <cuda_runtime.h>
#include <functional>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

inline float benchmark_kernel(std::function<void()> kernel_fn,
                               int warmup = 3,
                               int runs   = 10) {
    for (int i = 0; i < warmup; i++) kernel_fn();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < runs; i++) kernel_fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return total_ms / runs;
}

// RTX 3090 peaks: 35580 GFLOPS FP32, 936 GB/s
static constexpr float PEAK_GFLOPS    = 35580.0f;
static constexpr float PEAK_BANDWIDTH = 936.0f;

inline void print_metrics(const char* label, int N, float ms) {
    long long flops = 2LL * N * N * N;
    long long bytes = 3LL * N * N * sizeof(float);
    float gflops    = (float)flops / (ms * 1e6f);
    float bandwidth = (float)bytes / (ms * 1e6f);

    float compute_util  = gflops    / PEAK_GFLOPS    * 100.0f;
    float bandwidth_util = bandwidth / PEAK_BANDWIDTH * 100.0f;

    const char* bound;
    if (compute_util < 2.0f && bandwidth_util < 2.0f)
        bound = "latency-bound";
    else if (bandwidth_util > compute_util)
        bound = "memory-bound";
    else
        bound = "compute-bound";

    printf("%-20s | N=%-5d | %7.3f ms | %6.1f GB/s | %7.1f GFLOPS | %s\n",
           label, N, ms, bandwidth, gflops, bound);
}
