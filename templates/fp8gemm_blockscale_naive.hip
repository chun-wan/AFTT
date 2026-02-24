/*
 * Naive FP8 BlockScale GEMM - Deliberately Unoptimized
 *
 * C[M,N] = dequant(A[M,K], scale_A) @ dequant(B[K,N], scale_B)
 *
 * FP8 E4M3 FNUZ format with per-block scaling (block_size=128).
 *   scale_A[M, K/128]  — per-row, per-K-block
 *   scale_B[K/128, N/128] — per-K-block, per-N-block
 *
 * Anti-patterns included for optimization training:
 * 1. Scalar loads (1 byte at a time for FP8 data)
 * 2. Small 16x16 shared memory tiles
 * 3. FP8→F32 conversion per element in inner loop
 * 4. Block scale looked up inside inner K loop (redundant)
 * 5. Excessive __syncthreads barriers
 * 6. No loop unrolling
 * 7. Each thread computes only 1 output element
 */

#include <hip/hip_runtime.h>

#define TILE_K 16

__device__ __forceinline__ float fp8e4m3fnuz_to_f32(unsigned char val) {
    if (val == 0 || val == 0x80) return 0.0f;
    unsigned int sign = (val >> 7) & 1;
    unsigned int exp  = (val >> 3) & 0xF;
    unsigned int man  = val & 0x7;
    float result = ldexpf(1.0f + man * 0.125f, (int)exp - 8);
    return sign ? -result : result;
}

__global__ void fp8gemm_blockscale_naive_kernel(
    float* __restrict__ output,
    const unsigned char* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_B,
    const int M, const int N, const int K,
    const int scale_k, const int scale_n)
{
    __shared__ float As[TILE_K][TILE_K];
    __shared__ float Bs[TILE_K][TILE_K];

    int row = blockIdx.y * TILE_K + threadIdx.y;
    int col = blockIdx.x * TILE_K + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int aCol = t * TILE_K + threadIdx.x;
        int bRow = t * TILE_K + threadIdx.y;

        // Scalar FP8 load + per-element conversion + per-element scale lookup
        float a_val = 0.0f;
        if (row < M && aCol < K) {
            unsigned char a_fp8 = A[row * K + aCol];
            a_val = fp8e4m3fnuz_to_f32(a_fp8);
            float sa = scale_A[row * scale_k + aCol / 128];
            a_val *= sa;
        }
        As[threadIdx.y][threadIdx.x] = a_val;

        __syncthreads();

        float b_val = 0.0f;
        if (bRow < K && col < N) {
            unsigned char b_fp8 = B[bRow * N + col];
            b_val = fp8e4m3fnuz_to_f32(b_fp8);
            float sb = scale_B[(bRow / 128) * scale_n + col / 128];
            b_val *= sb;
        }
        Bs[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }
}

void fp8gemm_blockscale_naive(
    float* output, const unsigned char* A, const unsigned char* B,
    const float* scale_A, const float* scale_B,
    int M, int N, int K, hipStream_t stream = 0)
{
    int scale_k = (K + 127) / 128;
    int scale_n = (N + 127) / 128;
    dim3 grid((N + TILE_K - 1) / TILE_K, (M + TILE_K - 1) / TILE_K);
    dim3 block(TILE_K, TILE_K);
    fp8gemm_blockscale_naive_kernel<<<grid, block, 0, stream>>>(
        output, A, B, scale_A, scale_B, M, N, K, scale_k, scale_n);
}
