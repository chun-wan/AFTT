/*
 * Standalone extraction of aiter's rms_norm_kernel for benchmarking.
 * Original: aiter/csrc/kernels/rmsnorm_kernels.cu (vLLM/AMD)
 *
 * Dependencies inlined: vec8_t, wave_reduce (via __shfl_xor), block_reduce.
 * Operates on float for direct comparison with AFTT templates.
 *
 * Build: hipcc -O3 --offload-arch=gfx942 -o aiter_rmsnorm aiter_rmsnorm_standalone.hip
 * Run:   ./aiter_rmsnorm [num_tokens] [hidden_size]
 */
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

static constexpr int WARP_SIZE = 64;

template <typename scalar_t>
struct __align__(16) vec8_t {
    scalar_t x, y, z, w, u, v, s, t;

    __device__ vec8_t() : x(0), y(0), z(0), w(0), u(0), v(0), s(0), t(0) {}
    __device__ vec8_t(scalar_t x_, scalar_t y_, scalar_t z_, scalar_t w_,
                      scalar_t u_, scalar_t v_, scalar_t s_, scalar_t t_)
        : x(x_), y(y_), z(z_), w(w_), u(u_), v(v_), s(s_), t(t_) {}

    __device__ vec8_t operator*(const vec8_t &other) const {
        return vec8_t(x*other.x, y*other.y, z*other.z, w*other.w,
                      u*other.u, v*other.v, s*other.s, t*other.t);
    }

    __device__ vec8_t operator*(const float &scale) const {
        return vec8_t(x*scale, y*scale, z*scale, w*scale,
                      u*scale, v*scale, s*scale, t*scale);
    }

    __device__ void operator+=(const vec8_t &other) {
        x += other.x; y += other.y; z += other.z; w += other.w;
        u += other.u; v += other.v; s += other.s; t += other.t;
    }

    __device__ scalar_t sum() const { return x + y + z + w + u + v + s + t; }
};

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, WARP_SIZE);
    }
    return val;
}

template <typename T, int BlockSize>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static constexpr int num_warps = BlockSize / WARP_SIZE;
    __shared__ T smem[num_warps];

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane_id == WARP_SIZE - 1)
        smem[warp_id] = val;
    __syncthreads();

    // All threads participate: each reads per-warp sums and reduces
    // across num_warps values, matching aiter's block_reduce broadcast.
    val = smem[lane_id % num_warps];
    // Reduce within num_warps lanes using butterfly pattern
#pragma unroll
    for (int offset = num_warps / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset, WARP_SIZE);
    return val;
}

template <typename scalar_t, int BlockSize = 1024>
__global__ void aiter_rms_norm_kernel(
    scalar_t *__restrict__ out,
    const scalar_t *__restrict__ input,
    const scalar_t *__restrict__ weight,
    const float epsilon, const int num_tokens, const int hidden_size)
{
    vec8_t<scalar_t> v8_variance = {};

    auto *vectorized_out = reinterpret_cast<vec8_t<scalar_t> *>(out);
    auto const *vectorized_in = reinterpret_cast<vec8_t<scalar_t> const *>(input);
    auto const *vectorized_weight = reinterpret_cast<vec8_t<scalar_t> const *>(weight);
    const int vec_hidden_size = hidden_size >> 3;

#pragma unroll 4
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        vec8_t<scalar_t> x = vectorized_in[blockIdx.x * vec_hidden_size + idx];
        v8_variance += x * x;
    }

    float v8_sum = v8_variance.sum();
    float variance = block_reduce_sum<float, BlockSize>(v8_sum);

    float s_variance = rsqrtf(variance / hidden_size + epsilon);

#pragma unroll 4
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        vec8_t<scalar_t> v8_in = vectorized_in[blockIdx.x * vec_hidden_size + idx];
        vec8_t<scalar_t> v8_w = vectorized_weight[idx];
        vectorized_out[blockIdx.x * vec_hidden_size + idx] = v8_in * s_variance * v8_w;
    }
}

void rmsnorm_cpu_ref(float *out, const float *in, const float *w,
                     int num_tokens, int hidden_size, float eps) {
    for (int r = 0; r < num_tokens; ++r) {
        float ss = 0.f;
        for (int c = 0; c < hidden_size; ++c) {
            float v = in[r * hidden_size + c];
            ss += v * v;
        }
        float scale = 1.f / sqrtf(ss / hidden_size + eps);
        for (int c = 0; c < hidden_size; ++c)
            out[r * hidden_size + c] = in[r * hidden_size + c] * scale * w[c];
    }
}

int main(int argc, char **argv) {
    int num_tokens  = (argc > 1) ? atoi(argv[1]) : 128;
    int hidden_size = (argc > 2) ? atoi(argv[2]) : 8192;
    float epsilon   = 1e-6f;

    size_t N = (size_t)num_tokens * hidden_size;
    size_t bytes = N * sizeof(float);

    float *h_in  = (float *)malloc(bytes);
    float *h_w   = (float *)malloc(hidden_size * sizeof(float));
    float *h_out = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    srand(42);
    for (size_t i = 0; i < N; ++i)
        h_in[i] = (float)rand() / RAND_MAX * 2.f - 1.f;
    for (int i = 0; i < hidden_size; ++i)
        h_w[i] = (float)rand() / RAND_MAX * 0.5f + 0.75f;

    rmsnorm_cpu_ref(h_ref, h_in, h_w, num_tokens, hidden_size, epsilon);

    float *d_in, *d_w, *d_out;
    HIP_CHECK(hipMalloc(&d_in, bytes));
    HIP_CHECK(hipMalloc(&d_w, hidden_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, bytes));
    HIP_CHECK(hipMemcpy(d_in, h_in, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_w, h_w, hidden_size * sizeof(float), hipMemcpyHostToDevice));

    constexpr int BlockSize = 1024;
    dim3 grid(num_tokens);
    dim3 block(BlockSize);

    // Warmup
    for (int i = 0; i < 100; ++i) {
        aiter_rms_norm_kernel<float, BlockSize><<<grid, block>>>(
            d_out, d_in, d_w, epsilon, num_tokens, hidden_size);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Benchmark
    constexpr int NITER = 1000;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < NITER; ++i) {
        aiter_rms_norm_kernel<float, BlockSize><<<grid, block>>>(
            d_out, d_in, d_w, epsilon, num_tokens, hidden_size);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    float us_per_call = ms * 1000.f / NITER;

    size_t total_bytes = (size_t)num_tokens * hidden_size * sizeof(float) * 3;
    float bw = (float)total_bytes / (us_per_call * 1e-6) / 1e9;

    // Correctness
    HIP_CHECK(hipMemcpy(h_out, d_out, bytes, hipMemcpyDeviceToHost));
    float max_err = 0.f;
    for (size_t i = 0; i < N; ++i) {
        float err = fabsf(h_out[i] - h_ref[i]);
        if (err > max_err) max_err = err;
    }

    printf("RMSNorm aiter-standalone: %d tokens, hidden=%d\n", num_tokens, hidden_size);
    printf("  Time:       %.2f us\n", us_per_call);
    printf("  Bandwidth:  %.1f GB/s\n", bw);
    printf("  Max error:  %.2e\n", max_err);
    printf("  Correctness: %s\n", (max_err < 1e-3f) ? "PASS" : "FAIL");

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_in); hipFree(d_w); hipFree(d_out);
    free(h_in); free(h_w); free(h_out); free(h_ref);
    return (max_err < 1e-3f) ? 0 : 1;
}
