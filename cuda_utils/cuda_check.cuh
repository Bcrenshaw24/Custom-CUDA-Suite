/**
 * @file cuda_check.cuh
 * @brief Error-checking helpers for the CUDA runtime API.
 *
 * Two families of macros are provided:
 *  - CUDA_CHECK / CUDA_CHECK_LAST_ERROR: abort the process on failure. Use in
 *    applications, tests and benchmarks.
 *  - CUDA_RETURN_IF_ERROR: propagate the error to the caller. Use inside
 *    library code, which should never terminate the host process.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/**
 * @brief Evaluates a CUDA runtime call and aborts with a diagnostic on failure.
 *
 * @code
 * CUDA_CHECK(cudaMalloc(&d_ptr, bytes));
 * @endcode
 */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_check_err_ = (call);                             \
        if (cuda_check_err_ != cudaSuccess) {                                   \
            std::fprintf(stderr, "CUDA error %s (%s) at %s:%d\n  in: %s\n",     \
                         cudaGetErrorName(cuda_check_err_),                     \
                         cudaGetErrorString(cuda_check_err_),                   \
                         __FILE__, __LINE__, #call);                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

/**
 * @brief Checks for an error raised by the most recent kernel launch and
 *        aborts with a diagnostic on failure.
 *
 * Only catches launch-configuration errors; errors raised while the kernel
 * executes surface on the next synchronizing call.
 */
#define CUDA_CHECK_LAST_ERROR() CUDA_CHECK(cudaGetLastError())

/**
 * @brief Evaluates an expression yielding a cudaError_t and returns it from
 *        the enclosing function if it is not cudaSuccess.
 *
 * The enclosing function must return cudaError_t.
 */
#define CUDA_RETURN_IF_ERROR(call)                                              \
    do {                                                                        \
        const cudaError_t cuda_check_err_ = (call);                             \
        if (cuda_check_err_ != cudaSuccess) {                                   \
            return cuda_check_err_;                                             \
        }                                                                       \
    } while (0)

/**
 * @brief Integer ceiling division, usable on host and device.
 */
template <typename T>
__host__ __device__ constexpr T ceilDiv(T a, T b) {
    return (a + b - 1) / b;
}
