#pragma once 
#include <cuda_runtime.h>


#define CHECK_CUDA(call) 
do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__
        << " - " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE);
    }
} while(0)

void positional_encoding(float* input, int N, int d);

void weight(
    float* input, float* query, float* key, float* value, 
    float* query_output, float* key_output, float* value_output,
    int N, int D, int dk, int H); 

/* 
Input - N x D matrix 
Output - Q, K, V Blocks. H x N x dk

Basic GEMM mostly, output will be 
*/

