# Custom-CUDA-Suite

An open-source suite of custom CUDA libraries modeled on NVIDIA's official libraries (cuBLAS, cuDNN). Each routine is written from scratch, tested against a CPU reference, and benchmarked against its NVIDIA counterpart.

## Contents

| Library | Routine | Description | Status |
|---|---|---|---|
| cuBLAS | `basicGEMM` | Naive GEMM, one thread per output element | Done |
| cuBLAS | `tiledGEMM` | Shared-memory tiled GEMM (square matrices, C = A·B) | Done |
| cuBLAS | `regTiledGEMM` | 2D register-tiled GEMM | Done |
| cuBLAS | `wmmaGEMM` | Tensor Core GEMM via WMMA (fp16 inputs, fp32 accumulation) | Done (unoptimized) |
| cuDNN | Implicit GEMM convolution | Shared-memory tiled Implicit GEMM | Mostly Done |
| cuDNN | Basic attention (Non-fused) | Multiple kernel attention, forward pass | unfinished |

## Highlights

Measured on an NVIDIA T4 at M = N = K = 4096 (full results and analysis in [`cuBLAS/gemm/runs.md`](cuBLAS/gemm/runs.md)):

| Kernel | GFLOPS | Speedup vs naive | % of cuBLAS |
|---|---|---|---|
| `basicGEMM` | 472 | 1.0x | 12.0% |
| `tiledGEMM<32>` | 868 | 1.8x | 22.2% |
| `regTiledGEMM` | 3,062 | 6.5x | **78.2%** of `cublasSgemm` |
| `wmmaGEMM` | 3,141 | 6.7x | 13.6% of `cublasGemmEx` |

## Repository layout

```
cuBLAS/
  gemm/
    gemm.cuh        Public API
    gemm.cu         Kernels and launchers
    Makefile        Builds and runs tests/benchmarks
    runs.md         Benchmark results and analysis
    tests/
      test_gemm.cu  Correctness tests vs a CPU reference
      bench_gemm.cu Benchmarks vs cuBLAS
cuda_utils/
  cuda_check.cuh    CUDA_CHECK, CUDA_RETURN_IF_ERROR, ceilDiv
  cublas_check.cuh  CUBLAS_CHECK
```

## Requirements

- CUDA Toolkit 11.4 or newer (`nvcc`, cuBLAS for benchmarks)
- An NVIDIA GPU; `wmmaGEMM` requires compute capability 7.0+ (Volta or newer)
- `make` and a C++17 compiler

No local GPU? Everything runs on Google Colab's free T4 runtime (see below).

## Building and testing

From `cuBLAS/gemm`:

```bash
make test                          # correctness tests (exit status 1 on failure)
make bench                         # benchmarks at 512, 1024, 2048, 4096
make bench SIZES="1024 2048"       # custom sizes
make test ARCH=sm_80               # target a different GPU (default: sm_75)
```

### Google Colab

Select **Runtime → Change runtime type → T4 GPU**, then:

```python
!git clone https://github.com/Bcrenshaw24/Custom-CUDA-Suite.git
%cd /content/Custom-CUDA-Suite/cuBLAS/gemm
!make test
!make bench
```

## Usage

All routines compute `C = alpha * A * B + beta * C` on **row-major device pointers**, run asynchronously on the given stream, and return a `cudaError_t`.

```cpp
#include "cuBLAS/gemm/gemm.cuh"
#include "cuda_utils/cuda_check.cuh"

float *dA, *dB, *dC;  // M x K, K x N, M x N, allocated with cudaMalloc

// C = A * B on the default stream
CUDA_CHECK(regTiledGEMM(dA, dB, dC, M, N, K));

// C = 2 * A * B + C on a custom stream
CUDA_CHECK(regTiledGEMM(dA, dB, dC, M, N, K, 2.0f, 1.0f, stream));
CUDA_CHECK(cudaStreamSynchronize(stream));
```

Compile `cuBLAS/gemm/gemm.cu` alongside your own sources:

```bash
nvcc -std=c++17 -O3 -arch=sm_75 my_app.cu cuBLAS/gemm/gemm.cu -o my_app
```

Conventions shared by every routine:

- **Errors:** `cudaErrorInvalidValue` for bad arguments, `cudaErrorNotSupported` for inputs outside a routine's supported domain, otherwise the kernel launch status.
- **beta = 0:** C is write-only and may be uninitialized.
- **Templates:** templated routines are explicitly instantiated in `gemm.cu`; add an instantiation there to use another configuration.

See the docstrings in [`gemm.cuh`](cuBLAS/gemm/gemm.cuh) for each routine's requirements.
