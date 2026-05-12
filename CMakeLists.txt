cmake_minimum_required(VERSION 3.18)
project(cuda_matmul)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

add_executable(cuda_matmul
        src/main.cu
        src/matmul_cpu.cpp
        src/kernels/1_naive.cu
        src/kernels/2_tiled.cu
        src/kernels/3_blocked.cu
        src/kernels/4_blockedv2.cu
        src/kernels/5_vec4.cu
        include/benchmark.cuh
        include/matmul_cpu.h
        include/kernels.cuh
)

target_include_directories(cuda_matmul PRIVATE include)