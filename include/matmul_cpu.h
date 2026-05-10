#pragma once

void matmul_cpu(const float* a, const float* b, float* c, int N);
bool verify(const float* ref, const float* gpu, int N);
