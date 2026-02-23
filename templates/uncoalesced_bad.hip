#include <hip/hip_runtime.h>

// Anti-pattern: uncoalesced global memory access (column-major access on row-major data)
__global__ void uncoalesced_kernel(const float* input, float* output, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < height) {
        float sum = 0.0f;
        // BAD: stride-width access pattern is uncoalesced
        for (int j = 0; j < width; ++j) {
            sum += input[tid + j * height];
        }
        output[tid] = sum;
    }
}
