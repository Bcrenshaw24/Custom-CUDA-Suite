/**
 * @file common.cuh
 * @brief Shared helpers for the GEMM tests and benchmarks.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../../cuda_utils/cuda_check.cuh"

/** Function-pointer types matching the launchers in gemm.cuh. */
using GemmF32 = cudaError_t (*)(const float*, const float*, float*,
                                int, int, int, float, float, cudaStream_t);
using GemmF16 = cudaError_t (*)(const half*, const half*, float*,
                                int, int, int, float, float, cudaStream_t);

/**
 * @brief Owning wrapper around a cudaMalloc allocation (aborts on failure).
 */
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, bytes()));
        }
    }
    ~DeviceBuffer() { cudaFree(ptr_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    /** Copies @p host (which must hold count() elements) to the device. */
    void upload(const std::vector<T>& host) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
        }
    }

    /** Copies the buffer back to a new host vector. */
    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ > 0) {
            CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
        }
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

/** @brief Returns @p count floats drawn uniformly from [-1, 1]. */
inline std::vector<float> randomMatrix(size_t count, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(count);
    for (float& v : values) {
        v = dist(rng);
    }
    return values;
}

/** @brief Converts each element to half precision. */
inline std::vector<half> toHalf(const std::vector<float>& values) {
    std::vector<half> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = __float2half(values[i]);
    }
    return out;
}

/** @brief Rounds each element through half precision and back. */
inline std::vector<float> roundThroughHalf(const std::vector<float>& values) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = __half2float(__float2half(values[i]));
    }
    return out;
}

/**
 * @brief CPU reference GEMM, accumulating in double precision.
 *
 * Returns alpha * A * B + beta * C0 (row-major). C0 is ignored when beta == 0.
 */
inline std::vector<float> referenceGemm(const std::vector<float>& A,
                                        const std::vector<float>& B,
                                        const std::vector<float>& C0,
                                        int M, int N, int K,
                                        float alpha, float beta) {
    std::vector<float> C(static_cast<size_t>(M) * N);
    std::vector<double> row(N);
    for (int i = 0; i < M; ++i) {
        std::fill(row.begin(), row.end(), 0.0);
        for (int k = 0; k < K; ++k) {
            const double a = A[static_cast<size_t>(i) * K + k];
            const float* b_row = &B[static_cast<size_t>(k) * N];
            for (int j = 0; j < N; ++j) {
                row[j] += a * b_row[j];
            }
        }
        for (int j = 0; j < N; ++j) {
            const size_t idx = static_cast<size_t>(i) * N + j;
            double c = static_cast<double>(alpha) * row[j];
            if (beta != 0.0f) {
                c += static_cast<double>(beta) * C0[idx];
            }
            C[idx] = static_cast<float>(c);
        }
    }
    return C;
}

/** Result of an element-wise comparison. */
struct CompareResult {
    bool ok = true;
    double max_abs_err = 0.0;   ///< Largest |got - ref| (NaN if any NaN was seen).
    size_t first_bad = 0;       ///< Index of the first out-of-tolerance element.
};

/**
 * @brief Checks |got - ref| <= atol + rtol * |ref| for every element.
 *        NaN or Inf in @p got always fails.
 */
inline CompareResult compareMatrices(const std::vector<float>& got,
                                     const std::vector<float>& ref,
                                     double atol, double rtol) {
    CompareResult r;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (std::isnan(diff) || diff > r.max_abs_err) {
            r.max_abs_err = diff;
        }
        if (!(diff <= atol + rtol * std::fabs(ref[i])) && r.ok) {
            r.ok = false;
            r.first_bad = i;
        }
    }
    return r;
}

/** @brief max|got - ref| / max|ref|; robust for large outputs. NaN-propagating. */
inline double normalizedError(const std::vector<float>& got, const std::vector<float>& ref) {
    double max_diff = 0.0;
    double max_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - ref[i]);
        if (std::isnan(diff)) {
            return diff;
        }
        max_diff = std::fmax(max_diff, diff);
        max_ref = std::fmax(max_ref, std::fabs(static_cast<double>(ref[i])));
    }
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}
