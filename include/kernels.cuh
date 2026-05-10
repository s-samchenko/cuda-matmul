#pragma once
#include <functional>

struct KernelEntry {
    const char* name;
    std::function<void(const float*, const float*, float*, int)> launch;
};

void matmul_naive_launch(const float* A, const float* B, float* C, int N);
void matmul_tiled_launch(const float* A, const float* B, float* C, int N);
void matmul_blocked_launch(const float* A, const float* B, float* C, int N);
