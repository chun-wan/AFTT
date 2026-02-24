/*
 * Naive RMSNorm HIP Kernel - Deliberately Unoptimized
 *
 * RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight
 *
 * Anti-patterns included for optimization training:
 * 1. Scalar (element-by-element) global memory loads - no vectorization
 * 2. Shared memory tree reduction - no warp-level primitives
 * 3. Two-pass over input data (variance pass + normalize pass)
 * 4. Separate sqrt + division instead of rsqrtf
 * 5. No loop unrolling pragmas
 * 6. Small block size (256) limiting occupancy potential
 * 7. Extra __syncthreads barriers
 */

#include <hip/hip_runtime.h>
#include <cmath>

__shared__ float smem_reduce[256];

__global__ void rmsnorm_naive_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const int hidden_size,
    const float epsilon)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    // Pass 1: compute sum of squares (scalar loads, no vectorization)
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];          // scalar load
        local_sum_sq += val * val;
    }

    // Tree reduction in shared memory (no warp shuffle)
    smem_reduce[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem_reduce[tid] += smem_reduce[tid + stride];
        }
        __syncthreads();                   // barrier every iteration
    }

    // Compute normalization factor (sqrt + division, not rsqrtf)
    float variance = smem_reduce[0] / (float)hidden_size;
    float norm_factor = 1.0f / sqrtf(variance + epsilon);  // anti-pattern: separate sqrt+div
    __syncthreads();                       // unnecessary extra barrier

    // Pass 2: normalize and scale (reads input again from global memory)
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = row_input[i];          // redundant global memory read
        row_output[i] = val * norm_factor * weight[i];  // scalar store
    }
}

// Host launcher
void rmsnorm_naive(float* output, const float* input, const float* weight,
                   int num_tokens, int hidden_size, float epsilon,
                   hipStream_t stream = 0)
{
    dim3 grid(num_tokens);
    dim3 block(256);                       // small block size
    rmsnorm_naive_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, hidden_size, epsilon);
}

// Test harness
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv)
{
    const int num_tokens = 128;
    const int hidden_size = 8192;
    const float epsilon = 1e-6f;
    const size_t data_bytes = (size_t)num_tokens * hidden_size * sizeof(float);
    const size_t weight_bytes = hidden_size * sizeof(float);

    float* h_input  = (float*)malloc(data_bytes);
    float* h_weight = (float*)malloc(weight_bytes);
    float* h_output = (float*)malloc(data_bytes);

    srand(42);
    for (int i = 0; i < num_tokens * hidden_size; i++)
        h_input[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < hidden_size; i++)
        h_weight[i] = 1.0f;

    float *d_input, *d_weight, *d_output;
    hipMalloc(&d_input, data_bytes);
    hipMalloc(&d_weight, weight_bytes);
    hipMalloc(&d_output, data_bytes);
    hipMemcpy(d_input, h_input, data_bytes, hipMemcpyHostToDevice);
    hipMemcpy(d_weight, h_weight, weight_bytes, hipMemcpyHostToDevice);

    // Warmup
    rmsnorm_naive(d_output, d_input, d_weight, num_tokens, hidden_size, epsilon);
    hipDeviceSynchronize();

    // Benchmark
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    const int iters = 100;

    hipEventRecord(start);
    for (int i = 0; i < iters; i++)
        rmsnorm_naive(d_output, d_input, d_weight, num_tokens, hidden_size, epsilon);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    float avg_us = ms * 1000.0f / iters;

    hipMemcpy(h_output, d_output, data_bytes, hipMemcpyDeviceToHost);

    // Verify first row
    float ref_sum_sq = 0.0f;
    for (int i = 0; i < hidden_size; i++)
        ref_sum_sq += h_input[i] * h_input[i];
    float ref_norm = 1.0f / sqrtf(ref_sum_sq / hidden_size + epsilon);
    float max_err = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        float expected = h_input[i] * ref_norm * h_weight[i];
        float err = fabsf(h_output[i] - expected);
        if (err > max_err) max_err = err;
    }

    size_t total_bytes = (size_t)num_tokens * hidden_size * sizeof(float) * 3 + weight_bytes;
    float bw_gbps = (float)total_bytes / (avg_us * 1e-6) / 1e9;

    printf("RMSNorm Naive: tokens=%d, hidden=%d\n", num_tokens, hidden_size);
    printf("  Time: %.2f us/iter\n", avg_us);
    printf("  Bandwidth: %.1f GB/s\n", bw_gbps);
    printf("  Max error: %.2e\n", max_err);
    printf("  Correctness: %s\n", max_err < 1e-4 ? "PASS" : "FAIL");

    hipFree(d_input); hipFree(d_weight); hipFree(d_output);
    free(h_input); free(h_weight); free(h_output);
    return 0;
}
