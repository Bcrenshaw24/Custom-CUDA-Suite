#pragma once
#include <cuda_runtime.h>


void launchIGEMM(const float *input, const float* weight, float *output, int N, int C, int H, int W, int K, int P, int Q)