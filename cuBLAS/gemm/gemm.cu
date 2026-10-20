#include "gemm.cuh"

/**
 * @brief Multiplies two matrices, A & B.
 *
 * @tparam BM Matrix A Block Height.
 * @tparam BN Matrix B Block Width.
 * @tparam BK Matrix A Block Width & Matrix B Block Height.
 * @tparam TM Thread Block Width.
 * @tparam TN Thread Block Height.
 * 
 * @param A Matrix A.
 * @param B Matrix B.
 * @param C Output Matrix.
 * @param M Number of rows in Matrix A.
 * @param K Number of columns in Matrix A & Rows in Matrix B.
 * @param N Number of columns in Matrix B.
 * @param alpha Scalar to multiply against A * B.
 * @param beta Scalar to multiply against C if initialized.
 */
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void blockTiledGEMM(const float* A, const float* B, float* C, const int M, const int K, const int N, const float alpha, const float beta) {

    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    __shared__ float s_A[BM * BK];
    __shared__ float s_B[BK * BN];

    //BN / TN = How many sub-blocks are in a tile along an axis
    const uint thread_row = threadIdx.x / (BN / TN); 
    const uint thread_col = threadIdx.y % (BN / TN); 
    const uint num_threads = (BM / TM) * (BN / TN);

    A += block_row * BM * K; 
    B += block_col * BN;
    C += block_row * BM * N + block_col * BN;

    float thread_results[TM * TN] = {0.0f};
    float register_m[TM] = {0.0f};
    float register_n[TN] = {0.0f};


    for (uint block_k = 0; block_k < K; block_k += BK) {
    
    #pragma unroll 
        for (uint load_a = 0; load_a < BM * BK; load_a += num_threads) { 
            uint load_idx = threadIdx.x + load_a;
            uint a_row = load_idx / BK; 
            uint a_col = load_idx % BK;
            if (a_row < M && a_col < K) {
                s_A[load_a] = A[a_row * K + a_col];
            } else {
                s_A[load_a] = 0.0f;
            }
        }

    #pragma unroll 
        for (uint load_b = 0; load_b < BK * BN; load_b += num_threads) { 
            uint load_idx = threadIdx.x + load_b; 
            uint b_row = load_idx / BN; 
            uint b_col = load_idx % BN; 
            if (b_row < K && b_col < N) {
                s_B[load_b] = B[b_row * N + b_col];
            } else {
                s_B[load_b] = 0.0f;
            }
        }
        __syncthreads();

        A += BK; 
        B += BK * N;

        for (uint dot_idx = 0; dot_idx < BK; ++dot_idx) { 
            for (uint i = 0; i < TM; ++i) { 
                register_m[i] = s_A[(thread_row * TM + i) * BK + dot_idx];
            }

            for (uint i = 0; i < TN; ++i) { 
                register_n[i] =  s_B[dot_idx * BN + thread_col * TN + i];
            }

            for (uint res_m = 0; res_m < TM; ++res_m) { 
                for (uint res_n = 0; res_n < TN; ++res_n) { 
                    thread_results[res_m * TM + res_n] += register_m[res_m] * register_n[res_n];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll 
    for (uint res_m = 0; res_m < TM; ++res_m) { 
    #pragma unroll 
        for (uint res_n = 0; res_n < TN; ++res_n) { 
            uint c_row = block_row * BM + (thread_row * TM + res_m);
            uint c_col = block_col * BN + (thread_col * TN + res_n);

             if (c_row < M && c_col < N) {
               const uint c_idx = (thread_row * TM + res_m) * N + (thread_col * TN + res_n);
                C[c_idx] = alpha * thread_results[res_m * TN + res_n] + beta * C[c_idx];
            }
        }    
    }
}

/**
 * @brief Multiplies two matrices, A & B, using a block-tiled approach.
 * 
 * Computes the BLAS GEMM operation: C = (alpha * A * B) + (beta * C)
 *
 * @tparam BM Matrix A Block Height (Default: 128)
 * @tparam BN Matrix B Block Width (Default: 128)
 * @tparam BK Matrix A Block Width & Matrix B Block Height (Default: 8)
 * @tparam TM Thread Matrix Block Height per thread (Default: 8)
 * @tparam TN Thread Matrix Block Width per thread (Default: 8)
 * 
 * @param A Pointer to Matrix A.
 * @param B Pointer to Matrix B.
 * @param C Pointer to Output Matrix C.
 * @param M Number of rows in Matrix A and Matrix C.
 * @param K Number of columns in Matrix A and rows in Matrix B.
 * @param N Number of columns in Matrix B and Matrix C.
 * @param alpha Scalar multiplier applied against the matrix product (A * B).
 * @param beta Scalar multiplier applied against the existing Matrix C data.
 * 
 * @note Supports numerous mathematical operations on A, B, and C. Also natively 
 *       supports rectangular matrices.
 *
 * ### Examples
 * 
 * **Example 1: Standard Multiplication (C = A * B)**
 * ```cpp
 * float a[] = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2
 * float b[] = {5.0f, 6.0f, 7.0f, 8.0f}; // 2x2
 * float c[] = {0.0f, 0.0f, 0.0f, 0.0f}; // 2x2
 * 
 * // alpha = 1.0f, beta = 0.0f clears old C values
 * launchGEMM(a, b, c, 2, 2, 2, 1.0f, 0.0f);
 * ```
 * 
 * **Example 2: In-Place Accumulation (C += A * B)**
 * ```cpp
 * float a[] = {1.0f, 2.0f, 3.0f, 4.0f}; 
 * float b[] = {5.0f, 6.0f, 7.0f, 8.0f}; 
 * float c[] = {1.0f, 1.0f, 1.0f, 1.0f}; // Pre-existing values
 * 
 * // alpha = 1.0f, beta = 1.0f adds results to C
 * launchGEMM(a, b, c, 2, 2, 2, 1.0f, 1.0f);
 * ```
 * 
 * **Example 3: Rectangular Matrices with Custom Scaling Constants (C = 2C + 2AB)**
 * ```cpp
 * int M = 2, K = 3, N = 4;
 * float alpha = 2.0f;
 * float beta = 2.0f;
 * 
 * float a[] = {1.0f, 2.0f, 3.0f, 
 *              4.0f, 5.0f, 6.0f}; // 2x3
 *              
 * float b[] = {1.0f, 2.0f, 3.0f, 4.0f,
 *              5.0f, 6.0f, 7.0f, 8.0f,
 *              9.0f, 0.0f, 1.0f, 2.0f}; // 3x4
 *              
 * float c[] = {1.0f, 1.0f, 1.0f, 1.0f,
 *              1.0f, 1.0f, 1.0f, 1.0f}; // 2x4 Output
 * 
 * // Runs with default template parameter values: <128, 128, 8, 8, 8>
 * launchGEMM(a, b, c, M, K, N, alpha, beta);
 * ```
 */
template <
    int BM = 128, 
    int BN = 128, 
    int BK = 8, 
    int TM = 8, 
    int TN = 8
>
void launchGEMM(const float* A, const float* B, float* C, const int M, const int K, const int N, const float alpha, const float beta) {
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(BN / TN, BM / TM);

    dim3 dimGrid((N + BN - 1) / BN, (N + BM) / BM);

    blockTiledGEMM<BM, BN, BK, TM, TN><<<dimGrid, blockDim>>>(d_A, d_B, d_C, M, K, N, alpha, beta);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}