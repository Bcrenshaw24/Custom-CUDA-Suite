#include "../helpers.cu"
#include<iostream>

template<typename t>
__global__ void axpyKernel(int N, t alpha, const t* X, t* Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] = alpha * X[i] + Y[i]; // y = cV_1 + V_2
    }
}

template<typename t>
void launchAXPY(int N, t alpha, const t* h_X, t* h_Y) {

    t *d_X, *d_Y; 
    size_t size = N * sizeof(t); 

    cudaMalloc((void**)&d_X, size); // dynamically allocating variables of type t pointer and then 
                                    // letting that variable represent information sent to pointer with size
    cudaMalloc((void**)&d_Y, size);
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice); // copying varibales to pointer "copies" from the arguments, and ensuring size
    cudaMemcpy(d_Y, h_Y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    axpyKernel<<<blocksPerGrid, threadsPerBlock>>>(N, alpha, d_X, d_Y);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
    }

    cudaMemcpy(h_Y, d_Y, size, cudaMemcpyDeviceToHost); // copying variable from kernel pointer to regular variable 

    cudaFree(d_X); // freeing memory and given size
    cudaFree(d_Y);

    printVector(d_Y, N); // helper function

}
