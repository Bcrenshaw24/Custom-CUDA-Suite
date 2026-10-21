#include "igemm.cuh"


/**
 * @brief Performs a convolution across a matrix.
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
 */
template <int TILE_SIZE, int R, int S>
__global__ void implicit_gemm(
    const float* __restrict__ input,  // Shape: [N, H, W, C]
    const float* __restrict__ weight, // Shape: [R, S, C, K]
    float* __restrict__ output,       // Shape: [N, P, Q, K] 
    int N, int C, int H, int W,       
    int K,                           
    int P, int Q                    
) {
    int tile_h = TILE_SIZE + R - 1;
    int tile_w = TILE_SIZE + S - 1;

    __shared__ float s_I[(TILE_SIZE + R - 1) * (TILE_SIZE + S - 1)];
    __shared__ float s_W[R * S];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
 
    //Each output pixel is a weighted sum of the surrounding pixels
    float accum = 0.0f;

    //Calculates output indicies
    int n = row / (P * Q);
    int res = row % (P * Q);
    int p = res / Q;
    int q = res % Q;

    //Local indicies
    int local_p = p % tile_h;
    int local_q = q % tile_w;


    //Chunking, tiling per channel
    for (int c = 0; c < C; ++c) {

        //Tiling input and weight matricies
        for (int r = 0; r < R; ++r) { 
          for (int s = 0; s < S; ++s) { 

            //Calculates input indicies
            int h = p + r; 
            int w = q + s; 
            int input_idx = n * (H * W * C) + h * (W * C) + w * C + c;
            
            if (n < N && h < H && w < W && c < C) {
              s_I[(local_p + r) * tile_w + (local_q + s)] = input[input_idx]; 
            }
            else { 
              s_I[(local_p + r) * tile_w + (local_q + s)] = 0.0f;
            }

            int weight_idx = r * (S * C * K) + s * (C * K) + c * K + col;

            if (c < C && col < K) {
              s_W[r * S + s] = weight[weight_idx];
            }
            else { 
              s_W[r * S + s] = 0.0f;
            }
          }

        }

        __syncthreads();

        for (int r = 0; r < R; ++r) { 
          for (int s = 0; s < S; ++s) { 
            accum += s_I[(local_p + r) * tile_w + (local_q + s)] * s_W[r * S + s];
          } 
        }

        __syncthreads();


    }

    int global_idx = n * (P * Q * K) + p * (Q * K) + q * K + col;
    if (n < N && p < P && q < Q and col < K) {
        output[global_idx] = 0 + accum;
    }

}

template <int TILE_SIZE, int R, int S>
void launchIGEMMSimple(
  const float *input, const float* weight, float *output, 
  int N, int C, int H, int W, 
  int K, int P, int Q
)
{
    float* d_I;
    float* d_W;
    float* d_O;

    size_t input_size = N * H * W * C * sizeof(float);
    size_t weight_size = R * S * C * K * sizeof(float);
    size_t output_size = N * P * Q * K * sizeof(float);

    cudaMalloc((void**)&d_I, input_size);
    cudaMalloc((void**)&d_W, weight_size);
    cudaMalloc((void**)&d_O, output_size);

    cudaMemcpy(d_I, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, weight, weight_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    int gridX = (K + dimBlock.x - 1) / dimBlock.x;
    int gridY = ((N * P * Q) + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(gridX, gridY);

    implicit_gemm<TILE_SIZE, R, S><<<dimGrid, dimBlock>>>(d_I, d_W, d_O, N, C, H, W, K, P, Q);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaMemcpy(output, d_O, output_size, cudaMemcpyDeviceToHost);

    cudaFree(d_I);
    cudaFree(d_W);
    cudaFree(d_O);
}

template <int TILE_SIZE, int R, int S>
void launchIGEMM(
  const float *input, const float* weight, float *output, 
  int N, int C, int H, int W, 
  int K, int P, int Q
)
{
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    int gridX = (K + dimBlock.x - 1) / dimBlock.x;
    int gridY = ((N * P * Q) + dimBlock.y - 1) / dimBlock.y;
    dim3 dimGrid(gridX, gridY);

    implicit_gemm<TILE_SIZE, R, S><<<dimGrid, dimBlock>>>(input, weight, output, N, C, H, W, K, P, Q);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}