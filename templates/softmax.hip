#include <hip/hip_runtime.h>
#include <float.h>

__shared__ float shared_max[256];
__shared__ float shared_sum[256];

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = shared_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(row_input[i] - row_max);
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = shared_sum[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = expf(row_input[i] - row_max) / row_sum;
    }
}
