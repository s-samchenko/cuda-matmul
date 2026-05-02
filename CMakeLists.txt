cmake_minimum_required(VERSION 3.18)
project(cuda_matmul)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_matmul
        src/matmul_naive.cu
)

target_include_directories(cuda_matmul PRIVATE include)