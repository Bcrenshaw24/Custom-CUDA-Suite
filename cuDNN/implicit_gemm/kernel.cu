/** 
 * Implicit GEMM (General Matrix Multiply).
 * 
 * Primarily used for dense convolutions. 
 * 
 * Similar to standard GEMM but skips copying over chunks of an image, 
 * computes local indicies directly from global indicies.
 * 
 * Note: Will implemnt tiling at a later time.
 * 
 */


 //Dynamic TILE_SIZE 
template <int TILE_SIZE>
__global__ void implicit_gemm(
    const float* __restrict__ input,  // Shape: [N, C, H, W]
    const float* __restrict__ weight, // Shape: [K, C, R, S]
    float* __restrict__ output,       // Shape: [N, K, P, Q] Different dim. due to padding in input
    int N, int C, int H, int W,       // Input dims
    int K, int R, int S,              // Filter dims
    int P, int Q                      // Output spatial dims
) {

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Each output pixel is a weighted sum of the surrounding pixels
    float accum = 0.0f;

    //Calculates input matricies indicies per pixel per image per batch
    int n = row / (P * Q);
    int rem = row % (P * Q);
    int p = rem / Q;
    int q = rem % Q;

    //Thread has to be within output indicies, threads can't be lower than 0
    if ((row < P * Q * N) && (col < K)) {

    for (int i = 0; i < C * R * S; ++i) {

        //Mapping kernel indicies
        int c = i / (R * S); 
        int res = i % (R * S);
        int r = res / S;
        int s = res % S;

        //Mapping input indicies ((r, s) act as an offset)
        int h = p + r;
        int w = q + s;

        if (h >= H || h < 0 || w >= W || w < 0) { 
            continue;
        }
    
        accum += (input[h][w] * weight[r][s]);
    }
}

    output[p][q] = accum;
}

/**
 * @brief Perfoms a convolution on N imagees.
 *
 * @param input Input Matrix of size N * C * H * W.
 * @param weight Weight Matrix of size K * C * R * S.
 * @param output Output Matrix of size N * K * P * Q.
 * @param N Number of images.
 * @param C Number of color channels.
 * @param H Height of input images.
 * @param W Width of input images. 
 * @param K Number of filters.
 * @param R Height of filters.
 * @param S Width of filters.
 * @param P Height of output image. 
 * @param Q Width of output image.
 * @param TILE_SIZE size of each tile .
 */
void launchIGEMM(const float *input, const float* weight, float *output, int N, int C, int H, int W, int K, int R, int S, int P, int Q, int TILE_SIZE)
{ 
    float* d_I, float* d_W, float* d_O;
    
    int input_size = N * C * H * W * sizeof(float); 
    int weight_size = R * S * sizeof(float); 
    int output_size = N * K * P * Q * sizeof(float); 

    cudaMalloc((void**)d_I, input_size);
    cudaMalloc((void**)d_W, weight_size);
    cudaMalloc((void**)d_O, output_size);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);

    dim3 dimGrid()

}