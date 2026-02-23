#include <hip/hip_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__shared__ float tile[TILE_DIM][TILE_DIM + 1];

__global__ void matrix_transpose(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int width, int height) {
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in = xIndex + yIndex * width;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + i) < height) {
            tile[threadIdx.y + i][threadIdx.x] = input[index_in + i * width];
        }
    }

    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;

    int index_out = xIndex + yIndex * height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < height && (yIndex + i) < width) {
            output[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}
