# GEMM Benchmark Runs

**Setup:** Google Colab, NVIDIA T4 (sm_75), square matrices (M = N = K), alpha = 1, beta = 0.
Produced with `make bench ARCH=sm_75 SIZES="512 1024 2048 4096"`.

- **% cuBLAS** compares each kernel to cuBLAS *at the same precision*: fp32 kernels vs `cublasSgemm`, `wmmaGEMM` vs `cublasGemmEx` (fp16 inputs, fp32 accumulation, Tensor Cores).
- **vs basic** is the speedup over `basicGEMM` across both tiers. The fp16 rows are not a like-for-like comparison with fp32: they use lower-precision inputs and dedicated Tensor Core hardware.
- **check** compares each kernel's output to its cuBLAS reference (normalized max error).

## Summary

| Kernel | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| `basicGEMM` | 15.6% | 13.1% | 11.4% | 12.0% |
| `tiledGEMM<16>` | 18.9% | 18.7% | 15.5% | 16.4% |
| `tiledGEMM<32>` | 31.9% | 21.3% | 21.1% | 22.2% |
| `regTiledGEMM<128,128,8,8,8>` | 55.1% | 62.5% | 72.5% | **78.2%** |
| `wmmaGEMM<16,16,16>` (vs `cublasGemmEx`) | 15.0% | 9.6% | 12.1% | 13.6% |

Percent of the same-precision cuBLAS routine.

## Interpretation

**Register tiling is the big win.** `regTiledGEMM` is 6.5x faster than `basicGEMM` at 4096 and reaches 78% of `cublasSgemm`. Its share of cuBLAS grows with size (55% → 62% → 72% → 78%). At 512 there are only 16 blocks of 128x128 for the T4's 40 SMs, so the GPU is under-occupied; cuBLAS picks a different tile configuration per shape, which is why it stays fast at small sizes. Smaller tile configurations (e.g. 64x64) would likely help the small sizes.

**Shared-memory tiling alone gives only 1.4–1.8x.** The T4's L1/L2 caches already capture much of the reuse the naive kernel would otherwise miss, so explicit shared-memory tiling helps less than textbook numbers suggest. The real bottleneck is that every multiply-add still needs two shared-memory reads, which register tiling removes.

**`wmmaGEMM` is limited by memory, not by the Tensor Cores.** It reaches only ~13% of `cublasGemmEx` and is roughly as fast as the fp32 `regTiledGEMM`, despite the T4's much higher fp16 Tensor Core throughput. Each warp loads its A/B fragments straight from global memory with no shared-memory reuse and computes a single 16x16 output tile. Next steps, roughly in order of payoff:
1. Stage A/B tiles in shared memory once per block and have all warps read from there.
2. Give each warp several output tiles (e.g. 2x4 accumulator fragments) to reuse each loaded fragment.
3. Pad shared-memory rows to avoid bank conflicts.

(`cp.async` is another common technique but requires sm_80+, so it isn't available on the T4.)

**Correctness.** Every kernel matches cuBLAS. The fp32 kernels all report the *same* error at each size because they accumulate along K in the same order and so produce bit-identical results; an error of 0.0 means cuBLAS happened to use the same order at that size.

**Caveat on noise.** The T4 is capped at 70 W and throttles under sustained load, so the large sizes are somewhat noisy. For final numbers, repeat each size and take the best result.

## Raw output

```
== M = N = K = 512 ==
kernel                       precision          ms     GFLOPS  vs basic   % cuBLAS  check
cublasSgemm                  fp32            0.091     2961.4      6.4x     100.0%  reference
basicGEMM                    fp32            0.581      462.1      1.0x      15.6%  ok (err 0.0e+00)
tiledGEMM<16>                fp32            0.480      559.8      1.2x      18.9%  ok (err 0.0e+00)
tiledGEMM<32>                fp32            0.284      944.9      2.0x      31.9%  ok (err 0.0e+00)
regTiledGEMM<128,128,8,8,8>  fp32            0.164     1632.3      3.5x      55.1%  ok (err 0.0e+00)
cublasGemmEx (fp16, TC)      fp16->fp32      0.014    19289.3     41.7x     100.0%  reference
wmmaGEMM<16,16,16>           fp16->fp32      0.093     2895.0      6.3x      15.0%  ok (err 9.6e-07)

== M = N = K = 1024 ==
kernel                       precision          ms     GFLOPS  vs basic   % cuBLAS  check
cublasSgemm                  fp32            0.510     4208.1      7.6x     100.0%  reference
basicGEMM                    fp32            3.884      552.9      1.0x      13.1%  ok (err 1.6e-06)
tiledGEMM<16>                fp32            2.730      786.8      1.4x      18.7%  ok (err 1.6e-06)
tiledGEMM<32>                fp32            2.399      895.1      1.6x      21.3%  ok (err 1.6e-06)
regTiledGEMM<128,128,8,8,8>  fp32            0.817     2630.0      4.8x      62.5%  ok (err 1.6e-06)
cublasGemmEx (fp16, TC)      fp16->fp32      0.065    32859.9     59.4x     100.0%  reference
wmmaGEMM<16,16,16>           fp16->fp32      0.678     3166.6      5.7x       9.6%  ok (err 0.0e+00)

== M = N = K = 2048 ==
kernel                       precision          ms     GFLOPS  vs basic   % cuBLAS  check
cublasSgemm                  fp32            4.149     4140.4      8.7x     100.0%  reference
basicGEMM                    fp32           36.251      473.9      1.0x      11.4%  ok (err 2.0e-06)
tiledGEMM<16>                fp32           26.793      641.2      1.4x      15.5%  ok (err 2.0e-06)
tiledGEMM<32>                fp32           19.703      871.9      1.8x      21.1%  ok (err 2.0e-06)
regTiledGEMM<128,128,8,8,8>  fp32            5.722     3002.6      6.3x      72.5%  ok (err 2.0e-06)
cublasGemmEx (fp16, TC)      fp16->fp32      0.710    24186.9     51.0x     100.0%  reference
wmmaGEMM<16,16,16>           fp16->fp32      5.885     2919.2      6.2x      12.1%  ok (err 0.0e+00)

== M = N = K = 4096 ==
kernel                       precision          ms     GFLOPS  vs basic   % cuBLAS  check
cublasSgemm                  fp32           35.080     3917.8      8.3x     100.0%  reference
basicGEMM                    fp32          291.335      471.8      1.0x      12.0%  ok (err 0.0e+00)
tiledGEMM<16>                fp32          213.467      643.8      1.4x      16.4%  ok (err 0.0e+00)
tiledGEMM<32>                fp32          158.259      868.4      1.8x      22.2%  ok (err 0.0e+00)
regTiledGEMM<128,128,8,8,8>  fp32           44.883     3062.1      6.5x      78.2%  ok (err 0.0e+00)
cublasGemmEx (fp16, TC)      fp16->fp32      5.956    23075.3     48.9x     100.0%  reference
wmmaGEMM<16,16,16>           fp16->fp32     43.761     3140.6      6.7x      13.6%  ok (err 0.0e+00)
```
