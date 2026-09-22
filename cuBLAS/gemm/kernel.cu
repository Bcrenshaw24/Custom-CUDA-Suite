#include <iostream>
#include <cuda_runtime.h>

/**
 * @brief Multiplies two matricies, A & B.
 *
 * @param A Matrix A.
 * @param B Matrix B.
 * @param C Output Matrix.
 * @param N Number of rows.
 *
 * @note Algorithm assumes A and B are square matricies
 */

template <int TILE_SIZE>
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int N) {

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

/**
 * @brief Multiplies two matricies, A & B.
 *
 * @param A Matrix A.
 * @param B Matrix B.
 * @param C Output Matrix.
 * @param N Number of rows.
 * @param TILE_SIZE size of each tile 
 *
 * ### Examples
 * ```cpp
 * int a[] = {{1, 2}, {3, 4}};
 * int b[] = {{5, 6}, {7, 8}};
 * int c[] = {{0, 0}, {0, 0}}; 
 * launchGEMM(a, b, c, 2, 2);
 * ```
 */
void launchGEMM(const float* A, const float* B, float* C, const int N, const int TILE_SIZE) {
    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

    matrixMulTiled<TILE_SIZE><<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}