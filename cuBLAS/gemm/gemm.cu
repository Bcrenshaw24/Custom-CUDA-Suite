/**
 * @file gemm.cu
 * @brief Kernels and host launchers for the routines declared in gemm.cuh.
 */
#include "gemm.cuh"
#include "../../cuda_utils/cuda_check.cuh"

#include <cstdint>
#include <mma.h>

namespace {

constexpr unsigned int kMaxGridDimYZ = 65535;

/**
 * @brief Validates the arguments common to every GEMM routine.
 *
 * @param[out] quick_return Set to true when there is no work to do
 *                          (M == 0 or N == 0).
 * @return cudaErrorInvalidValue for negative dimensions or null pointers to
 *         non-empty matrices, cudaSuccess otherwise.
 */
cudaError_t validateGemmArgs(const void* A, const void* B, const void* C,
                             int M, int N, int K, bool& quick_return) {
    quick_return = false;
    if (M < 0 || N < 0 || K < 0) {
        return cudaErrorInvalidValue;
    }
    if (M == 0 || N == 0) {
        quick_return = true;
        return cudaSuccess;
    }
    if (C == nullptr || (K > 0 && (A == nullptr || B == nullptr))) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

/** @brief Rejects grids whose y-dimension exceeds the hardware limit. */
cudaError_t validateGrid(const dim3& grid) {
    return grid.y > kMaxGridDimYZ ? cudaErrorInvalidValue : cudaSuccess;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/**
 * @brief Naive GEMM kernel: each thread computes one element of C.
 *
 * Expects a 2D launch where x indexes columns of C and y indexes rows, so
 * consecutive threads in a warp read consecutive elements of B and write
 * consecutive elements of C.
 */
__global__ void basicGEMMKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[static_cast<size_t>(row) * K + k] * B[static_cast<size_t>(k) * N + col];
    }

    const size_t c_idx = static_cast<size_t>(row) * N + col;
    C[c_idx] = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * C[c_idx];
}

/**
 * @brief Shared-memory tiled GEMM kernel for square matrices (C = A * B).
 *
 * Launch with a (TILE_SIZE, TILE_SIZE) block and a
 * (ceil(N / TILE_SIZE), ceil(N / TILE_SIZE)) grid.
 *
 * @tparam TILE_SIZE Edge length of the square tile and thread block.
 * @param A Device pointer to A (N x N, row-major).
 * @param B Device pointer to B (N x N, row-major).
 * @param C Device pointer to C (N x N, row-major).
 * @param N Dimension of all three matrices.
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
 * @brief 2D register-tiled GEMM kernel.
 *
 * @todo NOT YET IMPLEMENTED. The body is a no-op, so regTiledGEMM currently
 *       leaves C unchanged. Follow the TODO steps below.
 *
 * Launch configuration (set by regTiledGEMM, do not change one without the other):
 *  - 1D block of (BM / TM) * (BN / TN) threads. Only threadIdx.x is meaningful.
 *  - blockIdx.y selects a band of BM rows of C, blockIdx.x a band of BN columns.
 *
 * @tparam BM Rows of C per thread block.
 * @tparam BN Columns of C per thread block.
 * @tparam BK Depth of the K-slice staged in shared memory per iteration.
 * @tparam TM Rows of C per thread.
 * @tparam TN Columns of C per thread.
 *
 * @param A     Device pointer to A (M x K, row-major).
 * @param B     Device pointer to B (K x N, row-major).
 * @param C     Device pointer to C (M x N, row-major).
 * @param M     Rows of A and C.
 * @param N     Columns of B and C.
 * @param K     Columns of A and rows of B.
 * @param alpha Scalar applied to A * B.
 * @param beta  Scalar applied to the existing contents of C.
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void blockTiledGEMM(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               float alpha, float beta) {
    // Guiding principle: never move the A/B/C pointers. Always compute GLOBAL
    // (row, col) coordinates and index with them. Every bounds check must
    // compare a global coordinate against M, N or K.

    // TODO(1): Declare the shared-memory tiles as 2D arrays:
    //            __shared__ float s_A[BM][BK];
    //            __shared__ float s_B[BK][BN];
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // TODO(2): Compute this block's origin in C:
    //            block_row0 = blockIdx.y * BM;
    //            block_col0 = blockIdx.x * BN;
    //          and the thread count: num_threads = (BM / TM) * (BN / TN).
    int block_row0 = blockIdx.y * BM;
    int block_col0 = blockIdx.x * BN;
    int num_threads = (BM / TM) * (BN / TN);

    // TODO(3): Compute this thread's COMPUTE role (which TM x TN patch it owns):
    //            compute_row = threadIdx.y / (BN / TN);
    //            compute_col = threadIdx.x % (BN / TN);   // threadIdx.x, not .y
    //          It owns C rows   block_row0 + compute_row * TM + [0, TM)
    //          and     C cols   block_col0 + compute_col * TN + [0, TN).
    int compute_row = threadIdx.x / (BN / TN); 
    int compute_col = threadIdx.x % (BN / TN);

    // TODO(4): Declare per-thread registers:
    //            float acc[TM][TN] = {};   // 2D, so no TM/TN index mix-ups
    //            float a_reg[TM];
    //            float b_reg[TN];
    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    // TODO(5): Main loop: for (int k0 = 0; k0 < K; k0 += BK) {
    //
    //   (a) Cooperatively load s_A (BM x BK elements, num_threads threads):
    //         for (int i = threadIdx.x; i < BM * BK; i += num_threads) {
    //             int r = i / BK, c = i % BK;              // LOAD role, tile-local
    //             int g_row = block_row0 + r, g_col = k0 + c;  // global
    //             s_A[r][c] = (g_row < M && g_col < K) ? A[(size_t)g_row * K + g_col] : 0.0f;
    //         }
    //       Note the store goes to the tile-local index derived from i, not
    //       to the loop's starting offset.
    //
    //   (b) Same for s_B (BK x BN): r = i / BN, c = i % BN,
    //       g_row = k0 + r, g_col = block_col0 + c, bounds-check against K and N.
    //
    //   (c) __syncthreads();   // tiles fully loaded before anyone reads them

    for (int k = 0; k < K; k += BK) { 
        for (int i = threadIdx.x; i < BM * BK; i += num_threads) { 
            int r = i / BK, c = i % BK;
            int g_row = block_row0 + r, g_col = k + c; 
            s_A[r][c] = (g_row < M && g_col < K) ? A[(size_t)g_row * K + g_col] : 0.0f;
        }
        for (int i = threadIdx.x; i < BN * BK; i += num_threads) { 
            int r = i / BN, c = i % BN;
            int g_row = k + r, g_col = block_col0 + c; 
            s_B[r][c] = (g_row < K && g_col < N) ? B[(size_t)g_row * N + g_col] : 0.0f;
        }
        __syncthreads(); 

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) { 
            #pragma unroll
            for (int i = 0; i < TM; ++i) { 
                a_reg[i] = s_A[compute_row * TM + i][kk];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) { 
                b_reg[j] = s_B[kk][compute_col * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {  
                #pragma unroll
                for (int j = 0; j < TN; ++j) {  
                    acc[i][j] += a_reg[i] * b_reg[j]; 
                } 
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) { 
        #pragma unroll
        for (int j = 0; j < TN; ++j) { 
            int row = block_row0 + compute_row * TM + i; 
            int col = block_col0 + compute_col * TN + j;
            if (row < M && col < N) { 
                size_t idx = (size_t)row * N + col; 
                C[idx] =  (beta == 0.0f) ? alpha * acc[i][j] : alpha * acc[i][j] + beta * C[idx];
            }
        }
    }
}

/**
 * @brief WMMA Tensor Core GEMM kernel: each warp computes one
 *        WMMA_M x WMMA_N tile of C.
 *
 * Launch with a (32 * WARPS_X, WARPS_Y) block; warps tile C along rows in x
 * and along columns in y. Requires M, N, K to be multiples of the WMMA shape
 * and sm_70 or newer; the body compiles to nothing on older architectures.
 */
template <int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void wmmaGEMMKernel(const half* __restrict__ A,
                               const half* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K,
                               float alpha, float beta) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    using namespace nvcuda;

    // Warp-uniform: blockDim.x is a multiple of warpSize.
    const int warp_m = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warp_n = blockIdx.y * blockDim.y + threadIdx.y;
    const int c_row = warp_m * WMMA_M;
    const int c_col = warp_n * WMMA_N;
    if (c_row >= M || c_col >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, A + static_cast<size_t>(c_row) * K + k, K);
        wmma::load_matrix_sync(b_frag, B + static_cast<size_t>(k) * N + c_col, N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* c_tile = C + static_cast<size_t>(c_row) * N + c_col;
    if (beta != 0.0f) {
        wmma::load_matrix_sync(c_frag, c_tile, N, wmma::mem_row_major);
        #pragma unroll
        for (int i = 0; i < c_frag.num_elements; ++i) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < acc_frag.num_elements; ++i) {
            acc_frag.x[i] *= alpha;
        }
    }
    wmma::store_matrix_sync(c_tile, acc_frag, N, wmma::mem_row_major);
#endif
}

} // namespace

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

cudaError_t basicGEMM(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha, float beta,
                      cudaStream_t stream) {
    bool quick_return = false;
    CUDA_RETURN_IF_ERROR(validateGemmArgs(A, B, C, M, N, K, quick_return));
    if (quick_return) {
        return cudaSuccess;
    }

    const dim3 block(32, 8);
    const dim3 grid(ceilDiv<unsigned int>(N, block.x), ceilDiv<unsigned int>(M, block.y));
    CUDA_RETURN_IF_ERROR(validateGrid(grid));

    basicGEMMKernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

template <int TILE_SIZE>
cudaError_t tiledGEMM(const float* A, const float* B, float* C,
                      int M, int N, int K,
                      float alpha, float beta,
                      cudaStream_t stream) {
    static_assert(TILE_SIZE > 0 && TILE_SIZE * TILE_SIZE <= 1024,
                  "TILE_SIZE^2 must not exceed 1024 threads per block");

    bool quick_return = false;
    CUDA_RETURN_IF_ERROR(validateGemmArgs(A, B, C, M, N, K, quick_return));
    if (quick_return) {
        return cudaSuccess;
    }
    // matrixMulTiled only implements C = A * B for square matrices.
    if (M != N || N != K || alpha != 1.0f || beta != 0.0f) {
        return cudaErrorNotSupported;
    }

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid(ceilDiv<unsigned int>(N, TILE_SIZE), ceilDiv<unsigned int>(N, TILE_SIZE));
    CUDA_RETURN_IF_ERROR(validateGrid(grid));

    matrixMulTiled<TILE_SIZE><<<grid, block, 0, stream>>>(A, B, C, N);
    return cudaGetLastError();
}

template <int BM, int BN, int BK, int TM, int TN>
cudaError_t regTiledGEMM(const float* A, const float* B, float* C,
                         int M, int N, int K,
                         float alpha, float beta,
                         cudaStream_t stream) {
    static_assert(BM % TM == 0 && BN % TN == 0, "BM/BN must be divisible by TM/TN");
    static_assert((BM / TM) * (BN / TN) <= 1024, "Too many threads per block");
    static_assert((BM * BK + BK * BN) * sizeof(float) <= 48 * 1024,
                  "Static shared memory exceeds 48 KiB");

    bool quick_return = false;
    CUDA_RETURN_IF_ERROR(validateGemmArgs(A, B, C, M, N, K, quick_return));
    if (quick_return) {
        return cudaSuccess;
    }

    // The kernel maps blockIdx.y to rows of C and blockIdx.x to columns.
    const dim3 block((BM / TM) * (BN / TN));
    const dim3 grid(ceilDiv<unsigned int>(N, BN), ceilDiv<unsigned int>(M, BM));
    CUDA_RETURN_IF_ERROR(validateGrid(grid));

    blockTiledGEMM<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

template <int WMMA_M, int WMMA_N, int WMMA_K>
cudaError_t wmmaGEMM(const half* A, const half* B, float* C,
                     int M, int N, int K,
                     float alpha, float beta,
                     cudaStream_t stream) {
    static_assert((WMMA_M == 16 && WMMA_N == 16 && WMMA_K == 16) ||
                  (WMMA_M == 32 && WMMA_N == 8  && WMMA_K == 16) ||
                  (WMMA_M == 8  && WMMA_N == 32 && WMMA_K == 16),
                  "Unsupported WMMA shape for half inputs / float accumulator");

    bool quick_return = false;
    CUDA_RETURN_IF_ERROR(validateGemmArgs(A, B, C, M, N, K, quick_return));
    if (quick_return) {
        return cudaSuccess;
    }
    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        return cudaErrorInvalidValue;
    }
    // load/store_matrix_sync require 256-bit aligned pointers.
    constexpr std::uintptr_t kAlign = 32;
    if (reinterpret_cast<std::uintptr_t>(A) % kAlign != 0 ||
        reinterpret_cast<std::uintptr_t>(B) % kAlign != 0 ||
        reinterpret_cast<std::uintptr_t>(C) % kAlign != 0) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    int cc_major = 0;
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&device));
    CUDA_RETURN_IF_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
    if (cc_major < 7) {
        return cudaErrorNotSupported;
    }

    // 4 x 4 warps per block -> each block computes a (4*WMMA_M) x (4*WMMA_N) tile.
    constexpr int kWarpsX = 4;
    constexpr int kWarpsY = 4;
    const dim3 block(32 * kWarpsX, kWarpsY);
    const dim3 grid(ceilDiv<unsigned int>(M, WMMA_M * kWarpsX), ceilDiv<unsigned int>(N, WMMA_N * kWarpsY));
    CUDA_RETURN_IF_ERROR(validateGrid(grid));

    wmmaGEMMKernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Explicit instantiations (keep in sync with the list in gemm.cuh)
// ---------------------------------------------------------------------------

template cudaError_t tiledGEMM<16>(const float*, const float*, float*, int, int, int, float, float, cudaStream_t);
template cudaError_t tiledGEMM<32>(const float*, const float*, float*, int, int, int, float, float, cudaStream_t);

template cudaError_t regTiledGEMM<128, 128, 8, 8, 8>(const float*, const float*, float*, int, int, int, float, float, cudaStream_t);

template cudaError_t wmmaGEMM<16, 16, 16>(const half*, const half*, float*, int, int, int, float, float, cudaStream_t);
template cudaError_t wmmaGEMM<32, 8, 16>(const half*, const half*, float*, int, int, int, float, float, cudaStream_t);
template cudaError_t wmmaGEMM<8, 32, 16>(const half*, const half*, float*, int, int, int, float, float, cudaStream_t);
