/**
 * @file test_gemm.cu
 * @brief Correctness tests for every routine in gemm.cuh.
 *
 * Each GPU result is compared against a double-precision CPU reference.
 * When beta == 0, C is pre-filled with NaN so a kernel that reads C anyway
 * fails. Exits with status 1 if any test fails.
 *
 * Build and run: `make test` from cuBLAS/gemm.
 */
#include <cstdio>
#include <string>
#include <vector>

#include "../gemm.cuh"
#include "common.cuh"

namespace {

struct Case {
    int M, N, K;
    float alpha, beta;
};

int g_passed = 0;
int g_failed = 0;
int g_skipped = 0;

std::string describe(const Case& c) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "M=%-4d N=%-4d K=%-4d alpha=%-4g beta=%-4g",
                  c.M, c.N, c.K, c.alpha, c.beta);
    return buf;
}

std::string describe(const CompareResult& r) {
    char buf[96];
    if (r.ok) {
        std::snprintf(buf, sizeof(buf), "max_abs_err=%.3g", r.max_abs_err);
    } else {
        std::snprintf(buf, sizeof(buf), "max_abs_err=%.3g first_bad_idx=%zu",
                      r.max_abs_err, r.first_bad);
    }
    return buf;
}

void record(bool ok, const std::string& name, const std::string& what, const std::string& detail) {
    (ok ? g_passed : g_failed)++;
    std::printf("[%s] %-28s %s  %s\n", ok ? "PASS" : "FAIL", name.c_str(), what.c_str(), detail.c_str());
}

void skip(const std::string& name, const std::string& reason) {
    ++g_skipped;
    std::printf("[SKIP] %-28s %s\n", name.c_str(), reason.c_str());
}

/** NaN-fills C when beta == 0 (C must not be read); otherwise uploads C0. */
void initOutput(DeviceBuffer<float>& dC, const std::vector<float>& hC0, float beta) {
    if (beta == 0.0f) {
        CUDA_CHECK(cudaMemset(dC.get(), 0xFF, dC.bytes()));  // 0xFFFFFFFF is a NaN
    } else {
        dC.upload(hC0);
    }
}

// fp32 accumulation error grows roughly linearly with K for inputs in [-1, 1].
double f32Atol(int K) { return 1e-5 * (K + 1); }
double f16Atol(int K) { return 1e-4 * (K + 1); }

void testF32(const std::string& name, GemmF32 gemm, const Case& c) {
    const size_t size_a = static_cast<size_t>(c.M) * c.K;
    const size_t size_b = static_cast<size_t>(c.K) * c.N;
    const size_t size_c = static_cast<size_t>(c.M) * c.N;

    const auto hA = randomMatrix(size_a, 1);
    const auto hB = randomMatrix(size_b, 2);
    const auto hC0 = randomMatrix(size_c, 3);

    DeviceBuffer<float> dA(size_a), dB(size_b), dC(size_c);
    dA.upload(hA);
    dB.upload(hB);
    initOutput(dC, hC0, c.beta);

    const cudaError_t status = gemm(dA.get(), dB.get(), dC.get(), c.M, c.N, c.K, c.alpha, c.beta, nullptr);
    if (status != cudaSuccess) {
        record(false, name, describe(c), std::string("returned ") + cudaGetErrorName(status));
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto ref = referenceGemm(hA, hB, hC0, c.M, c.N, c.K, c.alpha, c.beta);
    const auto result = compareMatrices(dC.download(), ref, f32Atol(c.K), 1e-4);
    record(result.ok, name, describe(c), describe(result));
}

void testF16(const std::string& name, GemmF16 gemm, const Case& c) {
    const size_t size_a = static_cast<size_t>(c.M) * c.K;
    const size_t size_b = static_cast<size_t>(c.K) * c.N;
    const size_t size_c = static_cast<size_t>(c.M) * c.N;

    // The reference uses the same half-rounded inputs the GPU sees, so only
    // accumulation error is measured.
    const auto hA = roundThroughHalf(randomMatrix(size_a, 1));
    const auto hB = roundThroughHalf(randomMatrix(size_b, 2));
    const auto hC0 = randomMatrix(size_c, 3);

    DeviceBuffer<half> dA(size_a), dB(size_b);
    DeviceBuffer<float> dC(size_c);
    dA.upload(toHalf(hA));
    dB.upload(toHalf(hB));
    initOutput(dC, hC0, c.beta);

    const cudaError_t status = gemm(dA.get(), dB.get(), dC.get(), c.M, c.N, c.K, c.alpha, c.beta, nullptr);
    if (status != cudaSuccess) {
        record(false, name, describe(c), std::string("returned ") + cudaGetErrorName(status));
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto ref = referenceGemm(hA, hB, hC0, c.M, c.N, c.K, c.alpha, c.beta);
    const auto result = compareMatrices(dC.download(), ref, f16Atol(c.K), 1e-3);
    record(result.ok, name, describe(c), describe(result));
}

void expectStatus(const std::string& name, const std::string& what, cudaError_t got, cudaError_t want) {
    const std::string detail = std::string("got ") + cudaGetErrorName(got) + ", want " + cudaGetErrorName(want);
    record(got == want, name, what, detail);
}

void testArgumentValidation() {
    DeviceBuffer<float> f(1024);
    DeviceBuffer<half> h(1024);
    float* p = f.get();

    expectStatus("basicGEMM", "negative M", basicGEMM(p, p, p, -1, 4, 4), cudaErrorInvalidValue);
    expectStatus("basicGEMM", "M=0 with null pointers", basicGEMM(nullptr, nullptr, nullptr, 0, 4, 4), cudaSuccess);
    expectStatus("basicGEMM", "null A with K>0", basicGEMM(nullptr, p, p, 4, 4, 4), cudaErrorInvalidValue);
    expectStatus("basicGEMM", "null C", basicGEMM(p, p, nullptr, 4, 4, 4), cudaErrorInvalidValue);

    expectStatus("tiledGEMM<16>", "non-square", tiledGEMM<16>(p, p, p, 4, 8, 4), cudaErrorNotSupported);
    expectStatus("tiledGEMM<16>", "alpha != 1", tiledGEMM<16>(p, p, p, 4, 4, 4, 2.0f, 0.0f), cudaErrorNotSupported);
    expectStatus("tiledGEMM<16>", "beta != 0", tiledGEMM<16>(p, p, p, 4, 4, 4, 1.0f, 1.0f), cudaErrorNotSupported);

    expectStatus("wmmaGEMM<16,16,16>", "M not multiple of 16",
                 wmmaGEMM<16, 16, 16>(h.get(), h.get(), p, 17, 16, 16), cudaErrorInvalidValue);
    expectStatus("wmmaGEMM<16,16,16>", "K not multiple of 16",
                 wmmaGEMM<16, 16, 16>(h.get(), h.get(), p, 16, 16, 8), cudaErrorInvalidValue);
    expectStatus("wmmaGEMM<16,16,16>", "misaligned A",
                 wmmaGEMM<16, 16, 16>(h.get() + 1, h.get(), p, 16, 16, 16), cudaErrorInvalidValue);
}

} // namespace

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::printf("Device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    // General shapes: ragged sizes exercise every bounds check, K=0 checks
    // C = beta * C, and non-default alpha/beta check the epilogue.
    const std::vector<Case> general = {
        {1, 1, 1, 1.0f, 0.0f},
        {64, 64, 64, 1.0f, 0.0f},
        {256, 256, 256, 1.0f, 0.0f},
        {129, 257, 33, 1.0f, 0.0f},
        {17, 3, 250, 1.0f, 0.0f},
        {256, 256, 256, 1.5f, -0.5f},
        {129, 257, 33, 0.5f, 2.0f},
        {64, 64, 0, 1.0f, 0.5f},
    };
    // tiledGEMM only supports square C = A * B.
    const std::vector<Case> square = {
        {1, 1, 1, 1.0f, 0.0f},
        {64, 64, 64, 1.0f, 0.0f},
        {100, 100, 100, 1.0f, 0.0f},
        {256, 256, 256, 1.0f, 0.0f},
    };
    // wmmaGEMM: M, N multiples of 32 and K multiple of 16 fit every WMMA shape.
    const std::vector<Case> tensor = {
        {32, 32, 16, 1.0f, 0.0f},
        {64, 64, 64, 1.0f, 0.0f},
        {256, 256, 256, 1.0f, 0.0f},
        {96, 160, 48, 1.0f, 0.0f},
        {128, 64, 256, 1.5f, -0.5f},
    };

    for (const Case& c : general) testF32("basicGEMM", basicGEMM, c);
    for (const Case& c : square)  testF32("tiledGEMM<16>", tiledGEMM<16>, c);
    for (const Case& c : square)  testF32("tiledGEMM<32>", tiledGEMM<32>, c);
    for (const Case& c : general) testF32("regTiledGEMM<128,128,8,8,8>", regTiledGEMM<128, 128, 8, 8, 8>, c);

    if (prop.major >= 7) {
        for (const Case& c : tensor) testF16("wmmaGEMM<16,16,16>", wmmaGEMM<16, 16, 16>, c);
        for (const Case& c : tensor) testF16("wmmaGEMM<32,8,16>", wmmaGEMM<32, 8, 16>, c);
        for (const Case& c : tensor) testF16("wmmaGEMM<8,32,16>", wmmaGEMM<8, 32, 16>, c);
    } else {
        skip("wmmaGEMM", "requires compute capability >= 7.0");
    }

    std::printf("\n");
    testArgumentValidation();

    CUDA_CHECK(cudaGetLastError());
    std::printf("\n%d passed, %d failed, %d skipped\n", g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
