#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief Multiplies two matricies, A & B.
 * 
 * @param A Matrix A
 * @param B Matrix B
 * @param C Output Matrix 
 * @param N Number of rows 
 * 
 * @note Algorithm assumes A and B are square matricies 
 */

__global__ void matrixMulTiled(const float* A, const float* B, float* C, int N, int TILE_SIZE) {

    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0;

    for (int sub = 0; sub < (N + TILE_SIZE -1) / TILE_SIZE; ++sub) {

    if (row < N && (sub *  TILE_SIZE + tx) < N) {
        s_A[ty][tx] = A[row * N + sub * TILE_SIZE + tx];
    }
    else {
        s_A[ty][tx] = 0.0;
    }
    if (col < N && (sub * TILE_SIZE + ty) < N) {
        s_B[ty][tx] = B[(sub * TILE_SIZE + ty) * N + col];
    }
    else {
        s_B[ty][tx] = 0.0;
    }

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += s_A[ty][k] * s_B[k][tx];
    }
    __syncthreads();
    }

    if (row < N && col < N) {
    C[row * N + col] = sum;
    }

}
