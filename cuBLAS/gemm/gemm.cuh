/**
 * @file gemm.cuh
 * @brief General matrix-matrix multiplication (GEMM) routines.
 *
 * Every routine computes the BLAS level-3 operation
 *
 *     C = alpha * (A * B) + beta * C
 *
 * where A is M x K, B is K x N and C is M x N.
 *
 * Conventions shared by all routines:
 *  - All matrices are dense and stored in **row-major** order with leading
 *    dimensions K (A), N (B) and N (C).
 *  - A, B and C are **device** pointers. A and B must not alias C.
 *  - Work is enqueued asynchronously on @p stream; the call returns before the
 *    computation finishes. Synchronize the stream before reading C on the host.
 *  - When beta == 0, C is write-only: its prior contents are never read, so it
 *    may be uninitialized.
 *  - If M == 0 or N == 0 the call is a no-op and returns cudaSuccess.
 *  - Each routine returns cudaSuccess, cudaErrorInvalidValue for bad arguments,
 *    cudaErrorNotSupported for inputs outside the routine's supported domain,
 *    or the error reported by the kernel launch. Errors that occur while the
 *    kernel executes are reported by a later synchronizing call.
 *
 * Templated routines are explicitly instantiated in gemm.cu for the
 * configurations listed on each function. Add an instantiation there to use
 * another configuration.
 */
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * @brief Naive GEMM: one thread computes one element of C.
 *
 * Reads A and B directly from global memory with no data reuse. Intended as a
 * correctness reference and performance baseline.
 *
 * @param[in]     A      Device pointer to A (M x K, row-major).
 * @param[in]     B      Device pointer to B (K x N, row-major).
 * @param[in,out] C      Device pointer to C (M x N, row-major).
 * @param[in]     M      Rows of A and C.
 * @param[in]     N      Columns of B and C.
 * @param[in]     K      Columns of A and rows of B.
 * @param[in]     alpha  Scalar applied to A * B.
 * @param[in]     beta   Scalar applied to the existing contents of C.
 * @param[in]     stream CUDA stream to enqueue the work on (default stream if null).
 *
 * @return cudaSuccess on success, otherwise a CUDA error code.
 */
cudaError_t basicGEMM(const float* A,
                      const float* B,
                      float* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f,
                      cudaStream_t stream = nullptr);

/**
 * @brief Shared-memory tiled GEMM.
 *
 * Each TILE_SIZE x TILE_SIZE thread block stages tiles of A and B through
 * shared memory so each global element is loaded once per tile.
 *
 * @warning The underlying kernel only supports square matrices with
 *          alpha == 1 and beta == 0 (i.e. C = A * B, M == N == K). Any other
 *          input returns cudaErrorNotSupported.
 *
 * @tparam TILE_SIZE Edge length of the square tile and thread block.
 *                   Instantiated for 16 and 32.
 *
 * @param[in]  A      Device pointer to A (M x K, row-major).
 * @param[in]  B      Device pointer to B (K x N, row-major).
 * @param[out] C      Device pointer to C (M x N, row-major).
 * @param[in]  M      Rows of A and C.
 * @param[in]  N      Columns of B and C.
 * @param[in]  K      Columns of A and rows of B.
 * @param[in]  alpha  Scalar applied to A * B. Must be 1.
 * @param[in]  beta   Scalar applied to C. Must be 0.
 * @param[in]  stream CUDA stream to enqueue the work on (default stream if null).
 *
 * @return cudaSuccess on success, otherwise a CUDA error code.
 */
template <int TILE_SIZE = 16>
cudaError_t tiledGEMM(const float* A,
                      const float* B,
                      float* C,
                      int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f,
                      cudaStream_t stream = nullptr);

/**
 * @brief 2D register-tiled GEMM.
 *
 * Each thread block computes a BM x BN tile of C, staging BM x BK slices of A
 * and BK x BN slices of B through shared memory. Each thread accumulates a
 * TM x TN sub-tile in registers to cut shared-memory traffic.
 *
 * Launched with (BM / TM) * (BN / TN) threads per block.
 *
 * @warning Not yet implemented: the kernel is currently a stub and C is left
 *          unchanged. See the TODOs in blockTiledGEMM (gemm.cu).
 *
 * @tparam BM Rows of C computed per thread block.
 * @tparam BN Columns of C computed per thread block.
 * @tparam BK Depth of the K-slice staged in shared memory per iteration.
 * @tparam TM Rows of C computed per thread.
 * @tparam TN Columns of C computed per thread.
 *         Instantiated for <128, 128, 8, 8, 8>.
 *
 * @param[in]     A      Device pointer to A (M x K, row-major).
 * @param[in]     B      Device pointer to B (K x N, row-major).
 * @param[in,out] C      Device pointer to C (M x N, row-major).
 * @param[in]     M      Rows of A and C.
 * @param[in]     N      Columns of B and C.
 * @param[in]     K      Columns of A and rows of B.
 * @param[in]     alpha  Scalar applied to A * B.
 * @param[in]     beta   Scalar applied to the existing contents of C.
 * @param[in]     stream CUDA stream to enqueue the work on (default stream if null).
 *
 * @return cudaSuccess on success, otherwise a CUDA error code.
 */
template <int BM = 128, int BN = 128, int BK = 8, int TM = 8, int TN = 8>
cudaError_t regTiledGEMM(const float* A,
                         const float* B,
                         float* C,
                         int M, int N, int K,
                         float alpha = 1.0f, float beta = 0.0f,
                         cudaStream_t stream = nullptr);

/**
 * @brief Tensor Core GEMM via WMMA (half-precision inputs, float accumulation).
 *
 * Each warp computes one WMMA_M x WMMA_N tile of C using
 * nvcuda::wmma::mma_sync.
 *
 * Requirements (otherwise the call fails without launching):
 *  - Device compute capability >= 7.0 (cudaErrorNotSupported).
 *  - M % WMMA_M == 0, N % WMMA_N == 0, K % WMMA_K == 0 (cudaErrorInvalidValue).
 *  - A, B and C aligned to 32 bytes (cudaErrorInvalidValue);
 *    cudaMalloc allocations satisfy this.
 *
 * @tparam WMMA_M Rows of the WMMA tile.
 * @tparam WMMA_N Columns of the WMMA tile.
 * @tparam WMMA_K Depth of the WMMA tile.
 *         Valid half/float shapes: <16, 16, 16>, <32, 8, 16>, <8, 32, 16>;
 *         all three are instantiated.
 *
 * @param[in]     A      Device pointer to A (M x K, row-major, half).
 * @param[in]     B      Device pointer to B (K x N, row-major, half).
 * @param[in,out] C      Device pointer to C (M x N, row-major, float).
 * @param[in]     M      Rows of A and C.
 * @param[in]     N      Columns of B and C.
 * @param[in]     K      Columns of A and rows of B.
 * @param[in]     alpha  Scalar applied to A * B.
 * @param[in]     beta   Scalar applied to the existing contents of C.
 * @param[in]     stream CUDA stream to enqueue the work on (default stream if null).
 *
 * @return cudaSuccess on success, otherwise a CUDA error code.
 */
template <int WMMA_M = 16, int WMMA_N = 16, int WMMA_K = 16>
cudaError_t wmmaGEMM(const half* A,
                     const half* B,
                     float* C,
                     int M, int N, int K,
                     float alpha = 1.0f, float beta = 0.0f,
                     cudaStream_t stream = nullptr);
