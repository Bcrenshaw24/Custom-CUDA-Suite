#pragma once
#include <cuda_runtime.h>
//Exposes endpoints to other programs 


//Optimized GEMM; Tiled to registers to minimize S-RAM reads
void launchGEMM(const float* A, const float* B, float* C, const int M, const int K, const int N, const float alpha, const float beta); 
