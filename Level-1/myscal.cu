template<typename t>
__global__ void scalKernel(int N, t alpha, t* X) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        X[i] = alpha * X[i]; // c[a,b,c,d] for every value in vector
    }
}

template<typename t>
void launchSCAL(int N, t alpha, t* h_X) {
    t *d_X; 
    size_t size = N * sizeof(t); // calc size for allocations

    //allocating variables 
    cudaMalloc((void**)&d_X, size); 
    
    // copying variables 
    cudaMemcpy(d_X, h_X, size, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    scalKernel<<<blocksPerGrid, threadsPerBlock>>>(N, alpha, d_X);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << "\n";
    }

    // copying variable from kernel pointer back to regular host variable
    cudaMemcpy(h_X, d_X, size, cudaMemcpyDeviceToHost); 

    // freeing memory
    cudaFree(d_X); 

    printVector(h_X, N); 
}
