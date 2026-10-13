#include "kernel.cuh"

/**
 * @brief Multiplies a batch of square matrices: C[i] = A[i] * B[i].
 *
 * @param a_array    Device array of batchCount device pointers, each to an N x N matrix A.
 * @param b_array    Device array of batchCount device pointers, each to an N x N matrix B.
 * @param c_array    Device array of batchCount device pointers, each to an N x N output C.
 * @param N          Rows (= columns) of every matrix in the batch.
 * @param batchCount Number of matrices (= number of pointers in each array).
 *
 * @note assume square matrices 
 * @note The pointer arrays AND the matrices they point to must be in device memory.
 */
template <int TILE_SIZE>
__global__ void gemmBatchedKernel(const float* const* a_array,
                                  const float* const* b_array,
                                  float* const* c_array,
                                  int N, int batchCount) {

}

/**
 * @brief Asynchronous launcher, mirrors cublasSgemmBatched's role.
 *
 * Caller owns all allocation and copies. This function only launches.
 *
 * @param stream Stream to launch on (0 = default stream).
 * @return cudaError_t result of the launch check (not of kernel completion).
 */
cudaError_t launchGemmBatched(const float* const* a_array,
                              const float* const* b_array,
                              float* const* c_array,
                              int N, int batchCount,
                              cudaStream_t stream /* = 0 in the header */) {

   
}
