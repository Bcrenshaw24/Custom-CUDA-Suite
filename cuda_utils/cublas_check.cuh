/**
 * @file cublas_check.cuh
 * @brief Error-checking helper for the cuBLAS API.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

/**
 * @brief Evaluates a cuBLAS call and aborts with a diagnostic on failure.
 *
 * @code
 * CUBLAS_CHECK(cublasCreate(&handle));
 * @endcode
 */
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        const cublasStatus_t cublas_check_status_ = (call);                     \
        if (cublas_check_status_ != CUBLAS_STATUS_SUCCESS) {                    \
            std::fprintf(stderr, "cuBLAS error %s (%s) at %s:%d\n  in: %s\n",   \
                         cublasGetStatusName(cublas_check_status_),             \
                         cublasGetStatusString(cublas_check_status_),           \
                         __FILE__, __LINE__, #call);                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
