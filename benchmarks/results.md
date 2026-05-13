# Benchmark Results
> Hardware: RTX 3090 (Vast.ai) | Peak: 35.5 TFLOPS, 936 GB/s
> All times: average of 10 runs after 3 warmup runs

---

## Matmul

| Date       | Kernel | N | ms/run | GB/s | GFLOPS | vs Naive | Notes |
|------------|--------|---|--------|------|--------|----------|-------|
| 2026-05-03 | matmul_naive | 256  | 0.023 | 33.7 | 1437.2 | 1.00x | baseline |
| 2026-05-03 | matmul_naive | 512  | 0.130 | 24.2 | 2067.4 | 1.00x | baseline |
| 2026-05-03 | matmul_naive | 1024 | 0.960 | 13.1 | 2236.7 | 1.00x | baseline |
| 2026-05-03 | matmul_naive | 2048 | 7.785 |  6.5 | 2206.9 | 1.00x | baseline |
| 2026-05-03 | matmul_tiled | 256  | 0.019 | 40.9 | 1743.0 | 1.21x | tile_size=16 |
| 2026-05-03 | matmul_tiled | 512  | 0.103 | 30.5 | 2603.2 | 1.26x | tile_size=16 |
| 2026-05-03 | matmul_tiled | 1024 | 0.753 | 16.7 | 2853.7 | 1.27x | tile_size=16 |
| 2026-05-03 | matmul_tiled | 2048 | 5.928 |  8.5 | 2898.0 | 1.31x | tile_size=16 |
| 2026-05-10 | matmul_naive | 256  | 0.024 | 32.6 | 1388.8 | 1.00x | baseline re-run |
| 2026-05-10 | matmul_naive | 512  | 0.134 | 23.5 | 2007.3 | 1.00x | baseline re-run |
| 2026-05-10 | matmul_naive | 1024 | 0.992 | 12.7 | 2165.8 | 1.00x | baseline re-run |
| 2026-05-10 | matmul_naive | 2048 | 7.822 |  6.4 | 2196.4 | 1.00x | baseline re-run |
| 2026-05-10 | matmul_tiled | 256  | 0.017 | 46.3 | 1974.0 | 1.41x | tile_size=16 |
| 2026-05-10 | matmul_tiled | 512  | 0.091 | 34.6 | 2955.4 | 1.47x | tile_size=16 |
| 2026-05-10 | matmul_tiled | 1024 | 0.661 | 19.0 | 3250.4 | 1.50x | tile_size=16 |
| 2026-05-10 | matmul_tiled | 2048 | 5.374 |  9.4 | 3196.9 | 1.46x | tile_size=16 |
| 2026-05-10 | matmul_blocked | 256  | 0.065 | 12.0 |  513.6 | 0.37x | BM=BN=128 BK=16 TM=TN=8 — latency-bound, too few blocks at small N |
| 2026-05-10 | matmul_blocked | 512  | 0.118 | 26.6 | 2271.7 | 1.14x | BM=BN=128 BK=16 TM=TN=8 |
| 2026-05-10 | matmul_blocked | 1024 | 0.276 | 45.6 | 7787.8 | 3.59x | BM=BN=128 BK=16 TM=TN=8 |
| 2026-05-10 | matmul_blocked | 2048 | 1.470 | 34.2 | 11685.0 | 5.32x | BM=BN=128 BK=16 TM=TN=8 |
| 2026-05-11 | matmul_blockedv2 | 256  | 0.061 | 12.9 |   549.8 | 0.40x | As transposed [BK][BM] for contiguous SMEM reads — latency-bound |
| 2026-05-11 | matmul_blockedv2 | 512  | 0.110 | 28.6 |  2443.1 | 1.22x | As transposed [BK][BM] for contiguous SMEM reads |
| 2026-05-11 | matmul_blockedv2 | 1024 | 0.253 | 49.8 |  8493.5 | 3.92x | As transposed [BK][BM] for contiguous SMEM reads |
| 2026-05-11 | matmul_blockedv2 | 2048 | 1.288 | 39.1 | 13340.7 | 6.07x | As transposed [BK][BM] for contiguous SMEM reads |
| 2026-05-12 | matmul_vec4 | 256  | 0.056 | 13.9 |   594.7 | 0.43x | float4 scatter As + float4 Bs — latency-bound |
| 2026-05-12 | matmul_vec4 | 512  | 0.100 | 31.3 |  2674.9 | 1.33x | float4 scatter As + float4 Bs |
| 2026-05-12 | matmul_vec4 | 1024 | 0.241 | 52.3 |  8927.9 | 4.10x | float4 scatter As + float4 Bs |
| 2026-05-12 | matmul_vec4 | 2048 | 1.134 | 44.4 | 15152.8 | 6.92x | float4 scatter As + float4 Bs |
| 2026-05-12 | matmul_doubleBuffering | 256  | 0.063 | 12.5 |   533.7 | 0.38x | cp.async Bs only, 32KB SMEM → 3 blocks/SM — latency-bound |
| 2026-05-12 | matmul_doubleBuffering | 512  | 0.113 | 27.8 |  2370.2 | 1.18x | cp.async Bs only, 32KB SMEM → 3 blocks/SM |
| 2026-05-12 | matmul_doubleBuffering | 1024 | 0.266 | 47.2 |  8063.3 | 3.71x | cp.async Bs only, 32KB SMEM → 3 blocks/SM |
| 2026-05-12 | matmul_doubleBuffering | 2048 | 1.399 | 36.0 | 12278.4 | 5.61x | cp.async Bs only, 32KB SMEM → 3 blocks/SM |
| 2026-05-12 | matmul_pipelined | 256  | 0.058 | 13.5 |   574.9 | 0.41x | As float4 scatter (LDS.128) + Bs cp.async + true double buf, 32KB → 3 blocks/SM — latency-bound |
| 2026-05-12 | matmul_pipelined | 512  | 0.104 | 30.3 |  2585.2 | 1.28x | As float4 scatter (LDS.128) + Bs cp.async + true double buf, 32KB → 3 blocks/SM |
| 2026-05-12 | matmul_pipelined | 1024 | 0.247 | 50.9 |  8691.1 | 3.99x | As float4 scatter (LDS.128) + Bs cp.async + true double buf, 32KB → 3 blocks/SM |
| 2026-05-12 | matmul_pipelined | 2048 | 1.332 | 37.8 | 12895.6 | 5.89x | As float4 scatter (LDS.128) + Bs cp.async + true double buf, 32KB → 3 blocks/SM |
