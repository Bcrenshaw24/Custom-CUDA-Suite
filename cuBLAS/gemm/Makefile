NVCC      ?= nvcc
ARCH      ?= sm_75
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(ARCH)
SIZES     ?=

BUILD   := build
LIB_SRC := gemm.cu
DEPS    := gemm.cu gemm.cuh tests/common.cuh ../../cuda_utils/cuda_check.cuh

.PHONY: all test bench clean

all: $(BUILD)/test_gemm $(BUILD)/bench_gemm

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_gemm: tests/test_gemm.cu $(DEPS) | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/test_gemm.cu $(LIB_SRC)

$(BUILD)/bench_gemm: tests/bench_gemm.cu $(DEPS) ../../cuda_utils/cublas_check.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ tests/bench_gemm.cu $(LIB_SRC) -lcublas

# Run correctness tests (exit status 1 on failure).
test: $(BUILD)/test_gemm
	./$(BUILD)/test_gemm

# Run benchmarks, e.g. `make bench SIZES="1024 2048"`.
bench: $(BUILD)/bench_gemm
	./$(BUILD)/bench_gemm $(SIZES)

clean:
	rm -rf $(BUILD)
