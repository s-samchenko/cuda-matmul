#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release > /dev/null
make -j$(nproc)
echo ""
./cuda_matmul
