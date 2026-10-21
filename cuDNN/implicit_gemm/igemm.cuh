#pragma once
#include <cuda_runtime.h>
#include <iostream> 


#define CHECK_CUDA(call) 
do { 
    cudaError_t err = call; 
    if (err != cudaSuccess) { 
        std::cerr << "CUDA Error in " << __FILE__ << ":" << __LINE__
        << " - " << cudaGetErrorString(err) << std::endl; 
        exit(EXIT_FAILURE);
    }
} while(0)


/**
 * @brief Performs a convolution across a matrix.
 * 
 * Simple Version. Assumes no hyperparameters & handles CUDA memory allocation natively.
 *
 * @tparam TILE_SIZE Size of Each Tile
 * @tparam R Height of Filter
 * @tparam S Width of Filter
 * 
 * @param input Input Matrix
 * @param weight Weight Matrix
 * @param output Output Matrix.
 * @param N Batch Size
 * @param C Number of Input Channels
 * @param H Input Height
 * @param W Output Height
 * @param K Number of Filters
 * @param P Output Height
 * @param Q Output Width
 * 
 * @note Assumes no additional hyperparameters and handles CUDA memory allocation natively.
 * 
 *  ### Examples
 * ```cpp
 *  // Input
 *  const int N = 1;
 *  const int H = 12;
 *  const int W = 12;
 *  const int C = 3;
 *
 *  // Weight
 *  const int K = 1;
 *  const int R = 2;
 *  const int S = 2;
 *
 *  // Output
 *  int P;
 *  int Q;
 * 
 * getOutputDimensions(H, W, R, S, P, Q); // P and Q are assigned values
 * 
 *
 *  size_t input_elements  = (size_t)N * H * W * C;
 *  size_t weight_elements = (size_t)K * R * S * C;   
 *  size_t output_elements = (size_t)N * K * P * Q;
 *
 *  std::vector<float> h_I(input_elements, 1.0f);
 *  std::vector<float> h_W(weight_elements, 2.0f);
 *  std::vector<float> h_O(output_elements, 0.0f);
 *  std::vector<float> h_O_cudnn(output_elements, 0.0f);
 *  launchIGEMM<16, R, S>(h_I.data(), h_W.data(), h_O.data(), N, C, H, W, K, P, Q);
 * ```
 */
void launchIGEMMSimple(
  const float *input, const float* weight, float *output, 
  int N, int C, int H, int W, 
  int K, int P, int Q
);

void launchIGEMM(
  const float *input, const float* weight, float *output, 
  int N, int C, int H, int W, 
  int K, int P, int Q
);