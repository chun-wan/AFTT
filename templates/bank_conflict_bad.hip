#include <hip/hip_runtime.h>

// Anti-pattern: LDS bank conflict due to stride-32 access
__shared__ float smem[32 * 32];

__global__ void bank_conflict_kernel(const float* input, float* output, int n) {
    int tid = threadIdx.x;
    // BAD: stride-32 causes all threads to hit same bank
    smem[tid * 32] = input[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    output[blockIdx.x * blockDim.x + tid] = smem[tid * 32];
}
