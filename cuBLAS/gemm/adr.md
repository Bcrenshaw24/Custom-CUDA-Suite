# Optimization 1 - Warm Up Runs 

Context: After Benchmarking, I realized that most of the time was spent on memory allocation. CUDA spends time spinnig up the neccesary drivers and allocating memory slowing down excecution time.

Parameters 1: Naive Tiled GEMM; TILE_SIZE = 16; N = 1024; T4 Colab GPU; No Warm-Up Run.

Results: 

Total Excecution Time: 5,797,358 ns (5.80 ms)
GFLOPS: 370

Parameters 2: Naive Tiled GEMM; TILE_SIZE = 16; N = 1024; T4 Colab GPU; Warm-Up Run.

Results: 

Total Excecution Time: 3,788,281 ns (3.79 ms)
GFLOPS: 566.9

Consequences: 

Positive: Stabalizies benchmarking results, puts GPU in a more realistic scenario. 
Negative: Not much use outside of benchmarking. Won't make your code actually faster. 

# Optimization 2 - Register Level Tiling 

Context: Even when tiling into shared memory, threads still suffer a retrival time penalty creating a memory-bound bottleneck. It takes up to 30 clock cycles (10 nanoseconds) to read each number from shared memory. 

Solution: Assigning each thread to a sub-block of numbers within a tile allows it to store each number within its registers. This only takes up to 2 clock cycles (1 nanosecond) to retrieve numbers resulting in a 90% decrease of time taken. To put it into perspective, a tile size of 16x16 would take a single thread 2550 nanoseconds to read each number once. With a sub-block size of 8x8, it takes a thread under 640 nanoseconds to add all numbers it needs. 


Results: 

Achieved an average of 95% of NVIDIA's cuBLAS library over 10 runs across diverse rectangular matrix dimensions. 

First Implementation Excecution Time: 3,605,912 ns (3.61 ms), 595.5 GFLOPS
Optimized Kernel Excecution Time: 3,605,912 ns (0.91 ms), 2,367.8 GFLOPS

Consequences: 

Positive: Makes code much much faster 
Negative: Usage is a bit more complex