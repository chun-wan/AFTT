#include <hip/hip_runtime.h>

__shared__ float shared_mean[256];
__shared__ float shared_var[256];

__global__ void layernorm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ output,
                                 int N, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += row_in[i];
    }
    shared_mean[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_mean[tid] += shared_mean[tid + stride];
        __syncthreads();
    }
    float mean = shared_mean[0] / N;
    __syncthreads();

    float local_var = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    shared_var[tid] = local_var;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared_var[tid] += shared_var[tid + stride];
        __syncthreads();
    }
    float var = shared_var[0] / N;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}
