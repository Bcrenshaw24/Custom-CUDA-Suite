#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include "kernel.cuh"

int main() {
    const int M = 1200; 
    const int K = 600;  
    const int N = 1000; 

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<float> h_A(size_A, 1.0f);
    std::vector<float> h_B(size_B, 2.0f);
    std::vector<float> h_C(size_C, 0.0f);

    // =========================================================================
    // 1. BENCHMARK: Custom Kernel
    // =========================================================================
    launchGEMM<128, 128, 8, 8, 8>(h_A.data(), h_B.data(), h_C.data(), M, K, N, 1, 0);
    launchGEMM<128, 128, 8, 8, 8>(h_A.data(), h_B.data(), h_C.data(), M, K, N, 1, 0);

    std::fill(h_C.begin(), h_C.end(), 0.0f);

    // =========================================================================
    // 2. BENCHMARK: cuBLAS Kernel
    // =========================================================================
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size_A * sizeof(float));
    cudaMalloc((void**)&d_B, size_B * sizeof(float));
    cudaMalloc((void**)&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Warmup cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);

    cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Shape: M=" << M << ", N=" << N << ", K=" << K << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
