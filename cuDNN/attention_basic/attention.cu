#include "attention.cuh"


/** 
 * @brief Peforms positional encoding across embeddings.
 * 
 * PE(pos, i) = sin(pos /10000^(2i/d))
 * 
 * pos = embed num, i = dimension num
 * 
 * @param input Embedding matrix 
 * @param N Number of embeddings 
 * @param d dimension of model 
 * 
 */
__global__ void positional_encoding(float* input, int N, int d) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float inv = (2 * i) * (1.0f / d); 
    float res = powf(10000, inv)
    float encode = sinf(row * (1.0f / res)); 

    input[row * N + col] += encode;
}


/** 
 * @brief Computes Q, K, and V blocks for multi-head attention.
 * 
 * Performs matrix multiplication between embedding matrix and QKV matricies.
 * 
 * 
 * @param input - Embedding matrix [N, D]
 * @param query - Query Weight Matrix [H, D, dk]
 * @param key - Key Weight Matrix [H, D, dk]
 * @param value - Value Weight Matrix [H, D, dk]
 * @param query_output - Query Output Matrix [H, N, dk]
 * @param key_output - Key Output Matrix [H, N, dk]
 * @param value_output - Value Output Matrix [H, N, dk]
 * @param N - Number of Embeddings 
 * @param D - Model Dimension
 * @param dk - Weight Matrix Dimension 
 * @param H - Number of Heads
 */
template <int TILE_SIZE>
__global__ void weight_kernel(
    float* input, float* query, float* key, float* value, 
    float* query_output, float* key_output, float* value_output,
    int N, int D, int dk, int H 
) 
{ 


}




/** 
 * @brief Computes Q, K, and V blocks for multi-head attention.
 * 
 * Performs matrix multiplication between embedding matrix and QKV matricies.
 * 
 * 
 * @param input Embedding matrix [N, D]
 * @param query Query Weight Matrix [H, D, dk]
 * @param key Key Weight Matrix [H, D, dk]
 * @param value Value Weight Matrix [H, D, dk]
 * @param query_output Query Output Matrix [H, N, dk]
 * @param key_output Key Output Matrix [H, N, dk]
 * @param value_output Value Output Matrix [H, N, dk]
 * @param N Number of Embeddings 
 * @param D Model Dimension
 * @param dk Weight Matrix Dimension 
 * @param H Number of Heads
 */
template <int TILE_SIZE>
void void weight(
    float* input, float* query, float* key, float* value, 
    float* query_output, float* key_output, float* value_output,
    int N, int D, int dk, int H 
) 
{ 
    dim3 dimBlock(TILE_SIZE, TILE_SIZE); 
    int gridX = ((H * D) + TILE_SIZE - 1) / dimBlock.x; 
    int gridY = ((H * N) + TILE_SIZE - 1) / dimBlock.y;
    dim3 dimGrid(gridX, gridY);

    weight_kernel<TILE_SIZE><<<dimBlock, dimGrid>>>()


    
}

