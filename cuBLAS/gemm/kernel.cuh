#pragma once
#include <cuda_runtime.h>
//Exposes endpoints to other programs 

//Square Matricies API Signature
void launchGEMM(const float* A, const float* B, float* C, const int N, const int TILE_SIZE);

//Rectangular Matricies API Signature (M x K multiplied by K x N) NOT implemented yet
void launchGEMM(const float* A, const float* B, const float* C, int M, int K, int N); 
