/**
 * @file bench_gemm.cu
 * @brief Benchmarks every GEMM routine against its cuBLAS counterpart.
 *
 * Usage: bench_gemm [size ...]   (square M = N = K; default 512 1024 2048 4096)
 *
 * Kernels fall into two precision tiers, each with its own cuBLAS reference:
 *  - fp32:          basic / tiled / regTiled   vs cublasSgemm
 *  - fp16 -> fp32:  wmmaGEMM (Tensor Cores)    vs cublasGemmEx (fp16 in, fp32 compute)
 *
 * "vs basic" is the speedup over basicGEMM across both tiers, which shows the
 * overall progression. Comparing the two tiers to each other is not
 * like-for-like: the fp16 tier uses lower-precision inputs and dedicated
 * Tensor Core hardware.
 *
 * Every kernel's output is checked against its tier's cuBLAS result; a kernel
 * that produces wrong output is flagged WRONG, and its timing is meaningless.
 *
 * Build and run: `make bench` (or `make bench SIZES="1024 2048"`) from cuBLAS/gemm.
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "../gemm.cuh"
#include "../../../cuda_utils/cublas_check.cuh"
#include "common.cuh"

namespace {

// tiledGEMM only supports alpha = 1, beta = 0, so every kernel uses these.
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;
constexpr float kTargetMs = 200.0f;  // approximate time spent timing each kernel
constexpr double kMaxNormalizedError = 1e-3;

enum class Tier { FP32, FP16 };

struct Result {
    std::string name;
    Tier tier;
    bool is_cublas;
    float ms;
    std::string check;
};

/** Warms up, then times @p launch averaged over enough iterations to fill ~kTargetMs. */
float timeMs(const std::function<void()>& launch, cudaStream_t stream) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    launch();  // warm-up
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float probe_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(start, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&probe_ms, start, stop));

    const int iters = std::clamp(static_cast<int>(kTargetMs / std::max(probe_ms, 1e-3f)), 3, 200);

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / iters;
}

std::string checkOutput(const DeviceBuffer<float>& got, const DeviceBuffer<float>& ref) {
    const double err = normalizedError(got.download(), ref.download());
    char buf[48];
    std::snprintf(buf, sizeof(buf), "%s (err %.1e)", err <= kMaxNormalizedError ? "ok" : "WRONG", err);
    return buf;
}

double gflops(int n, float ms) {
    return 2.0 * n * n * static_cast<double>(n) / (ms * 1e6);
}

void benchmarkSize(int n, cublasHandle_t handle, cudaStream_t stream, bool has_tensor_cores) {
    const size_t count = static_cast<size_t>(n) * n;
    const auto hA = randomMatrix(count, 1);
    const auto hB = randomMatrix(count, 2);

    DeviceBuffer<float> dA(count), dB(count), dC(count), dRef32(count), dRef16(count);
    DeviceBuffer<half> dAh(count), dBh(count);
    dA.upload(hA);
    dB.upload(hB);
    dAh.upload(toHalf(hA));
    dBh.upload(toHalf(hB));

    std::vector<Result> results;

    // --- fp32 tier -------------------------------------------------------
    // cuBLAS is column-major; computing C^T = B^T * A^T yields row-major C.
    results.push_back({"cublasSgemm", Tier::FP32, true, timeMs([&] {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &kAlpha, dB.get(), n, dA.get(), n, &kBeta, dRef32.get(), n));
    }, stream), "reference"});

    auto runF32 = [&](const char* name, GemmF32 gemm) {
        const float ms = timeMs([&] {
            CUDA_CHECK(gemm(dA.get(), dB.get(), dC.get(), n, n, n, kAlpha, kBeta, stream));
        }, stream);
        results.push_back({name, Tier::FP32, false, ms, checkOutput(dC, dRef32)});
    };
    runF32("basicGEMM", basicGEMM);
    runF32("tiledGEMM<16>", tiledGEMM<16>);
    runF32("tiledGEMM<32>", tiledGEMM<32>);
    runF32("regTiledGEMM<128,128,8,8,8>", regTiledGEMM<128, 128, 8, 8, 8>);

    // --- fp16 -> fp32 Tensor Core tier ----------------------------------
    if (has_tensor_cores && n % 16 == 0) {
        results.push_back({"cublasGemmEx (fp16, TC)", Tier::FP16, true, timeMs([&] {
            CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                      &kAlpha, dBh.get(), CUDA_R_16F, n, dAh.get(), CUDA_R_16F, n,
                                      &kBeta, dRef16.get(), CUDA_R_32F, n,
                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }, stream), "reference"});

        const float ms = timeMs([&] {
            CUDA_CHECK(wmmaGEMM<16, 16, 16>(dAh.get(), dBh.get(), dC.get(), n, n, n, kAlpha, kBeta, stream));
        }, stream);
        results.push_back({"wmmaGEMM<16,16,16>", Tier::FP16, false, ms, checkOutput(dC, dRef16)});
    }

    // --- report ----------------------------------------------------------
    float basic_ms = 0.0f;
    float cublas_ms[2] = {0.0f, 0.0f};
    for (const Result& r : results) {
        if (r.name == "basicGEMM") basic_ms = r.ms;
        if (r.is_cublas) cublas_ms[static_cast<int>(r.tier)] = r.ms;
    }

    std::printf("\n== M = N = K = %d ==\n", n);
    std::printf("%-28s %-10s %10s %10s %9s %10s  %s\n",
                "kernel", "precision", "ms", "GFLOPS", "vs basic", "% cuBLAS", "check");
    for (const Result& r : results) {
        const float ref_ms = cublas_ms[static_cast<int>(r.tier)];
        std::printf("%-28s %-10s %10.3f %10.1f %8.1fx %9.1f%%  %s\n",
                    r.name.c_str(),
                    r.tier == Tier::FP32 ? "fp32" : "fp16->fp32",
                    r.ms, gflops(n, r.ms),
                    basic_ms / r.ms,
                    100.0f * ref_ms / r.ms,
                    r.check.c_str());
    }
    if (!has_tensor_cores) {
        std::printf("(fp16 Tensor Core tier skipped: requires compute capability >= 7.0)\n");
    } else if (n % 16 != 0) {
        std::printf("(fp16 Tensor Core tier skipped: size must be a multiple of 16)\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) {
        const int n = std::atoi(argv[i]);
        if (n <= 0) {
            std::fprintf(stderr, "invalid size '%s'\n", argv[i]);
            return EXIT_FAILURE;
        }
        sizes.push_back(n);
    }
    if (sizes.empty()) {
        sizes = {512, 1024, 2048, 4096};
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("'%% cuBLAS' compares each kernel to cuBLAS at the same precision.\n");
    std::printf("'vs basic' compares against basicGEMM; fp16 rows are not like-for-like.\n");

    cudaStream_t stream;
    cublasHandle_t handle;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    // Default math mode: SGEMM stays true fp32 (no TF32 on Ampere+).
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    for (int n : sizes) {
        benchmarkSize(n, handle, stream, prop.major >= 7);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}
